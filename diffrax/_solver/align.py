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
    beta: Array
    a1: Array
    b1: Array
    aa: Array
    chh: Array


class _SolverState(eqx.Module):
    h: RealScalarLike
    taylor_coeffs: _Coeffs
    coeffs: _Coeffs
    rho: Array
    f: VF


# CONCERNING COEFFICIENTS:
# The coefficients used in a step of ALIGN depend on
# the time increment h, and the parameters gamma and u.
# Assuming the modelled SDE stays the same (i.e. gamma and u are fixed),
# then these coefficients must be recomputed each time h changes.
# Furthermore, for very small h, directly computing the coefficients
# via the function below can cause large floating point errors.
# Hence, we pre-compute the Taylor expansion of the ALIGN coefficients
# around h=0. Then we can compute the ALIGN coefficients either via
# the Taylor expansion, or via direct computation.
# In short the Taylor coefficients give a Taylor expansion with which
# one can compute the ALIGN coefficients more precisely for a small h.


def _directly_compute_coeffs(h, gamma, u):
    # compute the coefficients directly (as opposed to via Taylor expansion)
    dtype = jnp.dtype(gamma)
    al = gamma * h
    beta = jnp.exp(-al)
    a1 = (1 - beta) / gamma
    b1 = (beta + al - 1) / (gamma * al)
    aa = a1 / h

    al2 = al**2
    chh = 6 * (beta * (al + 2) + al - 2) / (al2 * gamma)

    out = _Coeffs(
        beta=beta,
        a1=a1,
        b1=b1,
        aa=aa,
        chh=chh,
    )

    return jtu.tree_map(lambda leaf: jnp.asarray(leaf, dtype=dtype), out)


def _tay_cfs_single(c, u) -> _Coeffs:
    # c is gamma
    dtype = jnp.dtype(c)
    c2 = jnp.square(c)
    c3 = c2 * c
    c4 = c3 * c
    c5 = c4 * c

    beta = jnp.array([1, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120], dtype=dtype)
    a1 = jnp.array([0, 1, -c / 2, c2 / 6, -c3 / 24, c4 / 120], dtype=dtype)
    b1 = jnp.array([0, 1 / 2, -c / 6, c2 / 24, -c3 / 120, c4 / 720], dtype=dtype)
    aa = jnp.array([1, -c / 2, c2 / 6, -c3 / 24, c4 / 120, -c5 / 720], dtype=dtype)
    chh = jnp.array([0, 1, -c / 2, 3 * c2 / 20, -c3 / 30, c4 / 168], dtype=dtype)

    return _Coeffs(
        beta=beta,
        a1=a1,
        b1=b1,
        aa=aa,
        chh=chh,
    )


def _comp_taylor_coeffs(gamma, u) -> _Coeffs:
    # When the step-size h is small the coefficients (which depend on h) need
    # to be computed via Taylor expansion to ensure numerical stability.
    # This precomputes the Taylor coefficients (depending on gamma and u), which
    # are then multiplied by powers of h, to get the coefficients of ALIGN.
    assert gamma.shape == u.shape

    if jnp.ndim(gamma) == 0:
        return _tay_cfs_single(gamma, u)

    return jax.vmap(_tay_cfs_single)(gamma, u)


def _eval_taylor(h, tay_cfs: _Coeffs) -> _Coeffs:
    # Multiplies the pre-computed Taylor coefficients by powers of h.
    # jax.debug.print("eval taylor for h = {h}", h=h)
    dtype = jnp.dtype(tay_cfs.a1)
    h_powers = jnp.power(h, jnp.arange(0, 6, dtype=dtype)).astype(dtype)
    return jtu.tree_map(
        lambda tay_leaf: jnp.tensordot(tay_leaf, h_powers, axes=1), tay_cfs
    )


class ALIGN(AbstractItoSolver):
    r"""The Adaptive Langevin via Interpolated Gradients and Noise method
    designed by James Foster. Only works for Underdamped Langevin Diffusion
    of the form

    $$d x_t = v_t dt$$

    $$d v_t = - gamma v_t dt - u ∇f(x_t) dt + (2γu)^(1/2) dW_t$$

    where $v$ is the velocity, $f$ is the potential, $γ$ is the friction, and
    $W$ is a Brownian motion.
    """

    term_structure = LangevinTerm
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: RealScalarLike = eqx.field(static=True)

    @staticmethod
    def minimal_levy_area():
        return "space-time"

    def __init__(self, taylor_threshold: RealScalarLike = 0.0):
        r"""**Arguments:**

        - `taylor_threshold`: If the product `h*gamma` is less than this, then
        the Taylor expansion will be used to compute the coefficients.
        Otherwise they will be computed directly. When using float32, the
        empirically optimal value is 0.1, and for float64 about 0.01.
        """
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 2

    def recompute_coeffs(
        self, h: RealScalarLike, gamma: Array, u: Array, tay_cfs: _Coeffs
    ):
        # Used when the step-size h changes and coefficients need to be recomputed
        # Depending on the size of h*gamma choose whether the Taylor expansion or
        # direct computation is more accurate.
        cond = h * gamma < self.taylor_threshold
        if jnp.ndim(gamma) == 0 and jnp.ndim(u) == 0:
            return lax.cond(
                cond,
                lambda h_: _eval_taylor(h_, tay_cfs),
                lambda h_: _directly_compute_coeffs(h_, gamma, u),
                h,
            )
        else:
            tay_out = _eval_taylor(h, tay_cfs)

            assert gamma.shape == u.shape

            def select_tay_or_direct(dummy):
                fun = lambda gam, _u: _directly_compute_coeffs(h, gam, _u)
                direct_out = vmap(fun)(gamma, u)
                return jtu.tree_map(
                    lambda tay_leaf, direct_leaf: jnp.where(
                        cond, tay_leaf, direct_leaf
                    ),
                    tay_out,
                    direct_out,
                )

            # If all entries of h*gamma are below threshold, only compute tay_out
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
        gamma, u, f = terms.args  # f is in fact grad(f)
        assert gamma.shape == u.shape
        h = t1 - t0

        tay_cfs = _comp_taylor_coeffs(gamma, u)
        coeffs = self.recompute_coeffs(h, gamma, u, tay_cfs)
        rho = jnp.sqrt(2 * gamma * u)

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
    ) -> tuple[tuple[Y, Y], tuple[Y, Y], DenseInfo, _SolverState, RESULTS]:
        del made_jump
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
            levy.H is not None
        ), "The Brownian motion must have levy_area='space-time'"
        _, w = levy.W
        _, hh = levy.H

        x0, v0 = y0
        assert x0.shape == v0.shape
        assert x0.ndim in [0, 1]

        assert jnp.shape(cfs.a1) == jnp.shape(gamma) or jnp.shape(cfs.a1) == jnp.shape(
            u
        )
        assert jnp.shape(gamma) in [(), x0.shape]

        uh = u * h
        f0 = st.f
        x1 = x0 + cfs.a1 * v0 - cfs.b1 * uh * f0 + st.rho * (cfs.b1 * w + cfs.chh * hh)
        f1 = f(x1)
        assert f1.shape == f0.shape, f"f0: {f0.shape}, f1: {f1.shape}"
        v1 = (
            cfs.beta * v0
            - u * ((cfs.a1 - cfs.b1) * f0 + cfs.b1 * f1)
            + st.rho * (cfs.aa * w - gamma * cfs.chh * hh)
        )

        y1 = (x1, v1)
        assert v1.dtype == x1.dtype == x0.dtype
        assert x1.shape == v1.shape == x0.shape

        error_estimate = (
            jnp.zeros_like(x0),
            -u * cfs.b1 * (f1 - f0),
        )

        dense_info = dict(y0=y0, y1=y1)
        st = _SolverState(
            h=h,
            taylor_coeffs=tay,
            coeffs=cfs,
            rho=st.rho,
            f=f1,
        )
        return y1, error_estimate, dense_info, st, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: tuple[Y, Y],
        args: PyTree,
    ) -> VF:
        return terms.vf(t0, y0, args)
