import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap
from jaxtyping import Array, PyTree

from .._custom_types import (
    BoolScalarLike,
    DenseInfo,
    LevyVal,
    RealScalarLike,
    VF,
    Y,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm, LangevinTerm
from .base import AbstractItoSolver


class _Coeffs(eqx.Module):
    beta_half: Array
    a_half: Array
    b_half: Array
    beta1: Array
    a1: Array
    b1: Array
    aa: Array
    chh: Array
    ckk: Array


class _SolverState(eqx.Module):
    h: RealScalarLike
    taylor_coeffs: _Coeffs
    coeffs: _Coeffs
    rho: Array
    f: VF


# CONCERNING COEFFICIENTS:
# The coefficients used in a step of ALIGN depend on
# the time increment h, and the parameters γ and u.
# Assuming the modelled SDE stays the same (i.e. γ and u are fixed),
# then these coefficients must be recomputed each time h changes.
# Furthermore, for very small h, directly computing the coefficients
# via the function below can cause large floating point errors.
# Hence, we pre-compute the Taylor expansion of the ALIGN coefficients
# around h=0. Then we can compute the ALIGN coefficients either via
# the Taylor expansion, or via direct computation.
# In short the Taylor coefficients give a Taylor expansion with which
# one can compute the ALIGN coefficients more precisely for a small h.


def _directly_compute_coeffs(h, γ, u):
    # compute the coefficients directly (as opposed to via Taylor expansion)
    # al = α
    al = γ * h
    dtype = jnp.dtype(γ)
    beta_half = jnp.exp(-al / 2)
    beta1 = jnp.exp(-al)
    a_half = (1 - beta_half) / γ
    a1 = (1 - beta1) / γ
    b_half = (beta_half + al / 2 - 1) / (γ * al)
    b1 = (beta1 + al - 1) / (γ * al)
    aa = a1 / h

    al2 = al**2
    al3 = al2 * al
    # chh_half = 3 * (beta_half * (8 + 4 * al) + al2 - 8) / (2 * al2 * γ)
    # ckk_half = 30 * (beta_half * (2 * al * (al + 6) + 24) + al2 - 24) / (al3 * γ)
    chh = 6 * (beta1 * (al + 2) + al - 2) / (al2 * γ)
    ckk = 60 * (beta1 * (al * (al + 6) + 12) - al * (al - 6) - 12) / (al3 * γ)

    out = _Coeffs(
        beta_half=beta_half,
        a_half=a_half,
        b_half=b_half,
        beta1=beta1,
        a1=a1,
        b1=b1,
        aa=aa,
        chh=chh,
        ckk=ckk,
    )

    return jtu.tree_map(lambda leaf: jnp.asarray(leaf, dtype=dtype), out)


def _tay_cfs_single(c, u) -> _Coeffs:
    # c is γ
    dtype = jnp.dtype(c)
    c2 = jnp.square(c)
    c3 = c2 * c
    c4 = c3 * c
    c5 = c4 * c

    beta_half = jnp.array(
        [1, -c / 2, c2 / 8, -c3 / 48, c4 / 384, -c5 / 3840], dtype=dtype
    )
    beta1 = jnp.array([1, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120], dtype=dtype)

    a_half = jnp.array([0, 1 / 2, -c / 8, c2 / 48, -c3 / 384, c4 / 3840], dtype=dtype)
    a1 = jnp.array([0, 1, -c / 2, c2 / 6, -c3 / 24, c4 / 120], dtype=dtype)
    # aa = a1/h
    aa = jnp.array([1, -c / 2, c2 / 6, -c3 / 24, c4 / 120, -c5 / 720], dtype=dtype)

    # b_half is not exactly b(1/2 h), but 1/2 * b(1/2 h)
    b_half = jnp.array(
        [0, 1 / 8, -c / 48, c2 / 384, -c3 / 3840, c4 / 46080], dtype=dtype
    )
    b1 = jnp.array([0, 1 / 2, -c / 6, c2 / 24, -c3 / 120, c4 / 720], dtype=dtype)

    # chh_half = jnp.array(
    #     [0, 1 / 2, -3 * c / 32, c2 / 80, -c3 / 768, c4 / 8960], dtype=dtype
    # )
    # ckk_half = jnp.array(
    #     [0, 15 / 8, -c / 2, 5 * c2 / 64, -c3 / 112, 5 * c4 / 6144], dtype=dtype
    # )
    chh = jnp.array([0, 1, -c / 2, 3 * c2 / 20, -c3 / 30, c4 / 168], dtype=dtype)
    ckk = jnp.array([0, 0, -c, c2 / 2, -c3 / 7, 5 * c4 / 168], dtype=dtype)

    return _Coeffs(
        beta_half=beta_half,
        a_half=a_half,
        b_half=b_half,
        beta1=beta1,
        a1=a1,
        b1=b1,
        aa=aa,
        chh=chh,
        ckk=ckk,
    )


def _comp_taylor_coeffs(γ, u) -> _Coeffs:
    # When the step-size h is small the coefficients (which depend on h) need
    # to be computed via Taylor expansion to ensure numerical stability.
    # This precomputes the Taylor coefficients (depending on γ and u), which
    # are then multiplied by powers of h, to get the coefficients of ALIGN.
    assert γ.shape == u.shape

    if jnp.ndim(γ) == 0:
        return _tay_cfs_single(γ, u)

    return jax.vmap(_tay_cfs_single)(γ, u)


def _eval_taylor(h, tay_cfs: _Coeffs) -> _Coeffs:
    # Multiplies the pre-computed Taylor coefficients by powers of h.
    # jax.debug.print("eval taylor for h = {h}", h=h)
    dtype = jnp.dtype(tay_cfs.a1)
    h_powers = jnp.power(h, jnp.arange(0, 6, dtype=dtype)).astype(dtype)
    return jtu.tree_map(
        lambda tay_leaf: jnp.tensordot(tay_leaf, h_powers, axes=1), tay_cfs
    )


class ShOULD(AbstractItoSolver):
    r"""The Shifted-ODE for Underdamped Langevin Diffusion method
    designed by James Foster. Only works for Underdamped Langevin Diffusion
    of the form

    $$d x_t = v_t dt$$

    $$d v_t = - γ v_t dt - u ∇f(x_t) dt + (2γu)^(1/2) dW_t$$

    where $v$ is the velocity, $f$ is the potential, $γ$ is the friction, and
    $W$ is a Brownian motion.
    """

    term_structure = LangevinTerm
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: RealScalarLike = eqx.field(static=True)

    @staticmethod
    def minimal_levy_area():
        return "space-time-time"

    def __init__(self, taylor_threshold: RealScalarLike = 0.0):
        r"""**Arguments:**

        - `taylor_threshold`: If the product `h*γ` is less than this, then
        the Taylor expansion will be used to compute the coefficients.
        Otherwise they will be computed directly. When using float32, the
        empirically optimal value is 0.1, and for float64 about 0.01.
        """
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 3

    def strong_order(self, terms):
        return 3

    def recompute_coeffs(self, h: RealScalarLike, γ: Array, u: Array, tay_cfs: _Coeffs):
        # Used when the step-size h changes and coefficients need to be recomputed
        # Depending on the size of h*γ choose whether the Taylor expansion or
        # direct computation is more accurate.
        cond = h * γ < self.taylor_threshold
        if jnp.ndim(γ) == 0 and jnp.ndim(u) == 0:
            return lax.cond(
                cond,
                lambda h_: _eval_taylor(h_, tay_cfs),
                lambda h_: _directly_compute_coeffs(h_, γ, u),
                h,
            )
        else:
            tay_out = _eval_taylor(h, tay_cfs)

            assert γ.shape == u.shape

            def select_tay_or_direct(dummy):
                fun = lambda gam, _u: _directly_compute_coeffs(h, gam, _u)
                direct_out = vmap(fun)(γ, u)
                return jtu.tree_map(
                    lambda tay_leaf, direct_leaf: jnp.where(
                        cond, tay_leaf, direct_leaf
                    ),
                    tay_out,
                    direct_out,
                )

            # If all entries of h*γ are below threshold, only compute tay_out
            # otherwise, compute both tay_out and direct_out and select the
            # correct one for each dimension
            return lax.cond(
                jnp.all(cond), lambda _: tay_out, select_tay_or_direct, None
            )

    def init(
        self,
        terms: LangevinTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Y, Y],
        args: PyTree,
    ) -> _SolverState:
        """Precompute _SolverState which carries the Taylor coefficients and the
        ALIGN coefficients (which can be computed from h and the Taylor coeffs).
        This method is FSAL, so _SolverState also carries the previous evaluation
        of grad_f.
        """
        γ, u, f = terms.args  # f is in fact grad(f)
        assert γ.shape == u.shape
        h = t1 - t0

        tay_cfs = _comp_taylor_coeffs(γ, u)
        coeffs = self.recompute_coeffs(h, γ, u, tay_cfs)
        rho = jnp.sqrt(2 * γ * u)

        x0, v0 = y0
        assert x0.shape == v0.shape
        assert x0.ndim in [0, 1]

        state_out = _SolverState(
            h=h,
            taylor_coeffs=tay_cfs,
            coeffs=coeffs,
            rho=rho,
            f=f(x0),
        )

        return state_out

    def step(
        self,
        terms: LangevinTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Y, Y],
        args: PyTree,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Y, Y], None, DenseInfo, _SolverState, RESULTS]:
        del made_jump, args
        st = solver_state
        h = t1 - t0
        gamma, u, f = terms.args

        h_state = st.h
        tay: _Coeffs = st.taylor_coeffs
        cfs = st.coeffs

        # If h changed recompute coefficients
        cond = jnp.isclose(h_state, h)
        cfs: _Coeffs = lax.cond(
            cond, lambda x: x, lambda _: self.recompute_coeffs(h, gamma, u, tay), cfs
        )
        # jax.debug.print("{h}", h=st['h'])

        drift, diffusion = terms.terms
        # compute the Brownian increment and space-time Levy area
        levy = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(levy, LevyVal)
        assert (
            levy.H is not None and levy.K is not None
        ), "The Brownian motion must have levy_area='space-time-time'"
        _, w = levy.W
        _, hh = levy.H
        _, kk = levy.K

        x0, v0 = y0
        assert x0.shape == v0.shape
        assert x0.ndim in [0, 1]

        assert jnp.shape(cfs.a1) == jnp.shape(gamma) or jnp.shape(cfs.a1) == jnp.shape(
            u
        )
        assert jnp.shape(gamma) in [(), x0.shape]
        uh = u * h
        f0 = st.f
        # z represents x^{(1)} in the paper
        z = (
            x0
            + cfs.a_half * (v0 + st.rho * (hh + 6.0 * kk))
            + cfs.b_half * (-uh * f0 + st.rho * (w - 12.0 * kk))
        )
        fz = f(z)
        # x_half = x0 + cfs.a_half * v0 - cfs.b_half * uh * (2*f0/3 + fz/3)
        # + st.rho * (cfs.a_half * w + cfs.chh_half * hh + cfs.ckk_half * kk)
        x_1 = (
            x0
            + cfs.a1 * v0
            - cfs.b1 * uh * (f0 / 3 + 2 * fz / 3)
            + st.rho * (cfs.b1 * w + cfs.chh * hh + cfs.ckk * kk)
        )
        f_1 = f(x_1)
        assert f_1.shape == f0.shape, (
            f"Shapes don't match." f" f0: {f0.shape}, f_1: {f_1.shape}"
        )
        v_1 = (
            cfs.beta1 * v0
            - uh * (cfs.beta1 / 6 * f0 + 2 / 3 * cfs.beta_half * fz + 1 / 6 * f_1)
            + st.rho * (cfs.aa * w - gamma * (cfs.chh * hh + cfs.ckk * kk))
        )

        y1 = (x_1, v_1)
        assert v_1.dtype == x_1.dtype == x0.dtype, (
            f"dtypes don't match. x0: {x0.dtype},"
            f" v_1: {v_1.dtype}, x_1: {x_1.dtype}"
        )
        assert x_1.shape == v_1.shape == x0.shape, (
            f"Shapes don't match. x0: {x0.shape},"
            f" v_1: {v_1.shape}, x_1: {x_1.shape}"
        )

        # TODO: compute error estimate

        dense_info = dict(y0=y0, y1=y1)
        st = _SolverState(
            h=h,
            taylor_coeffs=tay,
            coeffs=cfs,
            rho=st.rho,
            f=f_1,
        )
        return y1, None, dense_info, st, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: tuple[Y, Y],
        args: PyTree,
    ) -> VF:
        return terms.vf(t0, y0, args)
