from typing import Optional, Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, PyTree

from diffrax import (
    AbstractSpaceTimeLevyArea,
    LocalLinearInterpolation,
    AbstractFosterLangevinSRK,
)
from diffrax._term import (
    UnderdampedLangevinLeaf,
    UnderdampedLangevinX,
    RealScalarLike,
)
from diffrax._solver.foster_langevin_srk import (
    AbstractCoeffs,
    UnderdampedLangevinArgs,
)


# For an explanation of the coefficients, see foster_langevin_srk.py
class _UBUCoeffs(AbstractCoeffs):
    beta_half: PyTree[ArrayLike]
    a_half: PyTree[ArrayLike]
    b_half: PyTree[ArrayLike]
    beta1: PyTree[ArrayLike]
    a1: PyTree[ArrayLike]
    b1: PyTree[ArrayLike]
    aa: PyTree[ArrayLike]
    chh: PyTree[ArrayLike]
    chh_half: PyTree[ArrayLike]
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, beta_half, a_half, b_half, beta1, a1, b1, aa, chh, chh_half):
        self.beta_half = beta_half
        self.a_half = a_half
        self.b_half = b_half
        self.beta1 = beta1
        self.a1 = a1
        self.b1 = b1
        self.aa = aa
        self.chh = chh
        self.chh_half = chh_half
        all_leaves = jtu.tree_leaves(
            [
                self.beta_half,
                self.a_half,
                self.b_half,
                self.beta1,
                self.a1,
                self.b1,
                self.aa,
                self.chh,
                self.chh_half,
            ]
        )
        self.dtype = jnp.result_type(*all_leaves)


class UBU(AbstractFosterLangevinSRK[_UBUCoeffs, None]):
    r"""The UBU method by Alfonso Álamo Zapatero.
    This is a second order solver for the Underdamped Langevin Diffusion.
    Uses one vector field evaluation per step.

    ??? cite "Reference"

        ```bibtex
        @inproceedings{Zapatero2017WordSF,
          title={Word series for the numerical integration of stochastic differential equations},
          author={Alfonso {\'A}lamo Zapatero},
          year={2017},
          url={https://api.semanticscholar.org/CorpusID:125699606}
        }
        ```

    """

    interpolation_cls = LocalLinearInterpolation
    minimal_levy_area = AbstractSpaceTimeLevyArea
    taylor_threshold: float = eqx.field(static=True)
    _is_fsal = False

    def __init__(self, taylor_threshold: float = 0.1):
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
        return 2.0

    def _directly_compute_coeffs_leaf(
        self, h: RealScalarLike, c: UnderdampedLangevinLeaf
    ) -> _UBUCoeffs:
        del self
        # c is a leaf of gamma
        # compute the coefficients directly (as opposed to via Taylor expansion)
        al = c * h
        beta_half = jnp.exp(-al / 2)
        beta1 = jnp.exp(-al)
        a_half = (1 - beta_half) / c
        a1 = (1 - beta1) / c
        b_half = (beta_half + al / 2 - 1) / (al * c)
        b1 = (beta1 + al - 1) / (al * c)
        aa = a1 / h

        al2 = al**2
        chh = 6 * (beta1 * (al + 2) + al - 2) / (al2 * c)
        chh_half = 6 * (beta_half * (al + 2) + al2 / 4 - 2) / (al2 * c)

        return _UBUCoeffs(
            beta_half=beta_half,
            a_half=a_half,
            b_half=b_half,
            beta1=beta1,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
            chh_half=chh_half,
        )

    def _tay_coeffs_single(self, c: UnderdampedLangevinLeaf) -> _UBUCoeffs:
        del self
        # c is a leaf of gamma
        zero = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c

        # Coefficients of the Taylor expansion, starting from 5th power
        # to 0th power. The descending power order is because of jnp.polyval
        beta_half = jnp.stack(
            [-c5 / 3840, c4 / 384, -c3 / 48, c2 / 8, -c / 2, one], axis=-1
        )
        beta1 = jnp.stack([-c5 / 120, c4 / 24, -c3 / 6, c2 / 2, -c, one], axis=-1)
        a_half = jnp.stack(
            [c4 / 3840, -c3 / 384, c2 / 48, -c / 8, one / 2, zero], axis=-1
        )
        a1 = jnp.stack([c4 / 120, -c3 / 24, c2 / 6, -c / 2, one, zero], axis=-1)
        aa = jnp.stack([-c5 / 720, c4 / 120, -c3 / 24, c2 / 6, -c / 2, one], axis=-1)
        b_half = jnp.stack(
            [c4 / 46080, -c3 / 3840, c2 / 384, -c / 48, one / 8, zero], axis=-1
        )
        b1 = jnp.stack([c4 / 720, -c3 / 120, c2 / 24, -c / 6, one / 2, zero], axis=-1)
        chh = jnp.stack([c4 / 168, -c3 / 30, 3 * c2 / 20, -c / 2, one, zero], axis=-1)
        chh_half = jnp.stack(
            [c4 / 8960, -c3 / 768, c2 / 80, -3 * c / 32, one / 2, zero], axis=-1
        )

        correct_shape = jnp.shape(c) + (6,)
        assert (
            beta_half.shape
            == a_half.shape
            == b_half.shape
            == beta1.shape
            == a1.shape
            == b1.shape
            == aa.shape
            == chh.shape
            == chh_half.shape
            == correct_shape
        )

        return _UBUCoeffs(
            beta_half=beta_half,
            a_half=a_half,
            b_half=b_half,
            beta1=beta1,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
            chh_half=chh_half,
        )

    def _compute_step(
        self,
        h: RealScalarLike,
        levy: AbstractSpaceTimeLevyArea,
        x0: UnderdampedLangevinX,
        v0: UnderdampedLangevinX,
        underdamped_langevin_args: UnderdampedLangevinArgs,
        coeffs: _UBUCoeffs,
        rho: UnderdampedLangevinX,
        prev_f: Optional[UnderdampedLangevinX],
        args: PyTree[Any],
    ) -> tuple[
        UnderdampedLangevinX,
        UnderdampedLangevinX,
        UnderdampedLangevinX,
        None,
    ]:
        del prev_f
        dtypes = jtu.tree_map(jnp.result_type, x0)
        w: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.W, dtypes)
        hh: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.H, dtypes)

        gamma, u, f = underdamped_langevin_args

        z = (
            x0**ω
            + coeffs.a_half**ω * v0**ω
            + rho**ω * (coeffs.b_half**ω * w**ω + coeffs.chh_half**ω * hh**ω)
        ).ω
        fz = f(z, args)
        fz_uh = (fz**ω * u**ω * h).ω
        v1 = (
            coeffs.beta1**ω * v0**ω
            - coeffs.beta_half**ω * fz_uh**ω
            + rho**ω * (coeffs.aa**ω * w**ω - gamma**ω * coeffs.chh**ω * hh**ω)
        ).ω
        x1 = (
            x0**ω
            + coeffs.a1**ω * v0**ω
            - coeffs.a_half**ω * fz_uh**ω
            + rho**ω * (coeffs.b1**ω * w**ω + coeffs.chh**ω * hh**ω)
        ).ω

        return x1, v1, None, None
