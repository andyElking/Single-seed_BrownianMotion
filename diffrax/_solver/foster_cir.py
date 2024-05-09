from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import relu

from .._custom_types import (
    Args,
    BoolScalarLike,
    DenseInfo,
    LevyVal,
    RealScalarLike,
    VF,
    Y,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractStratonovichSolver


_ErrorEstimate: TypeAlias = Y
_SolverState: TypeAlias = None


class HOStS(AbstractStratonovichSolver):
    r"""Foster's High-Order Strang Splitting method for the
     Cox-Ingersoll-Ross process.
    It is unique in that it is tha only method that never
    becomes negative as long as $b > \frac{\sigma^2}{4 a}$.

    Based on equation $(5.3)$ from
    ??? cite "Reference"

        ```bibtex
        @misc{foster2023high,
          title={High order splitting methods for SDEs satisfying
            a commutativity condition},
          author={James Foster and Goncalo dos Reis and Calum Strange},
          year={2023},
          eprint={2210.17543},
          archivePrefix={arXiv},
          primaryClass={math.NA}
        ```
    """

    term_structure = MultiTerm[tuple[ODETerm, AbstractTerm]]
    interpolation_cls = LocalLinearInterpolation
    half_stepping: bool = eqx.field(static=True, default=False)
    negative_error_multiplier: RealScalarLike = eqx.field(static=True, default=0.0)

    @staticmethod
    def minimal_levy_area():
        return "space-time"

    # def __init__(
    #     self,
    #     negative_error_multiplier: RealScalarLike = 0.0,
    #     half_stepping: bool = False,
    # ):
    #     self.half_stepping = half_stepping
    #     self.negative_error_multiplier = negative_error_multiplier

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.4

    def init(
        self,
        terms: MultiTerm[tuple[ODETerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def _substep(
        self,
        terms: MultiTerm[tuple[ODETerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> tuple[Y, jax.Array]:
        assert isinstance(y0, jax.Array)
        assert y0.shape == ()
        a, b, sigma = args
        # we need to shift b by sigma^2/(4a) to get the Stratonovich form
        tilde_b = b - (sigma**2) / (4 * a)

        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        levy = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(levy, LevyVal)
        assert (
            levy.H is not None
        ), "The Brownian motion must have levy_area='space-time'"
        w = levy.W
        hh = levy.H
        assert w.shape == hh.shape == ()

        c1 = jnp.exp(a * dt * (np.sqrt(3) - 3.0) / 6.0)
        c2 = jnp.exp(-a * dt / np.sqrt(3))

        z1 = c1 * y0 + (1 - c1) * tilde_b
        z2 = jnp.square(jnp.sqrt(relu(z1)) + sigma / 2 * (w / 2 + np.sqrt(3) * hh))
        z3 = c2 * z2 + (1 - c2) * tilde_b
        z4 = jnp.square(jnp.sqrt(relu(z3)) + sigma / 2 * (w / 2 - np.sqrt(3) * hh))
        y1 = c1 * z4 + (1 - c1) * tilde_b

        stg = jnp.stack([z1, z2, z3, z4, y1], axis=0)
        assert stg.shape == (5,)

        return y1, stg

    def step(
        self,
        terms: MultiTerm[tuple[ODETerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        assert isinstance(y0, jax.Array)
        assert y0.shape == ()

        dt = t1 - t0
        y_coarse, stg_crs = self._substep(terms, t0, t1, y0, args)
        if self.half_stepping:
            t_half = t0 + dt / 2.0
            y_half, stg_half = self._substep(terms, t0, t_half, y0, args)
            y1, stg1 = self._substep(terms, t_half, t1, y_half, args)

            error = jnp.abs(y1 - y_coarse)
            if self.negative_error_multiplier > 0:
                all_stg = jnp.concatenate([stg_half, stg1, stg_crs], axis=0)
                error += self.negative_error_multiplier * relu(-jnp.min(all_stg))
        else:
            y1 = y_coarse
            error = None

        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)
