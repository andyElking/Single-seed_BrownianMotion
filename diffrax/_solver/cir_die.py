from typing import Optional
from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp

from .._custom_types import (
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    VF,
    Y,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractStratonovichSolver


_ErrorEstimate: TypeAlias = Optional[Y]
_SolverState: TypeAlias = Y


class DriftImplicitEulerCIR(AbstractStratonovichSolver):
    r"""The drift-implicit square-root Euler method for
    the Cox-Ingersoll-Ross process."""

    term_structure = MultiTerm[tuple[ODETerm, AbstractTerm]]
    interpolation_cls = LocalLinearInterpolation
    half_stepping: bool = eqx.field(static=True, default=False)

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 1

    def init(
        self,
        terms: MultiTerm[tuple[ODETerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        # We will carry y=sqrt(x) in the solver state and
        # output x=y^2 as the result of the scheme
        return jnp.sqrt(y0)

    def _substep(
        self,
        terms: MultiTerm[tuple[ODETerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> Y:
        a, b, sigma = args
        # we need to shift b by sigma^2/(4a) to get the Stratonovich form
        tilde_b = b - (sigma**2) / (4 * a)

        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        w = diffusion.contr(t0, t1)

        term1 = y0 + sigma / 2 * w
        term2 = 2.0 + a * dt

        root_term = jax.nn.relu(jnp.square(term1) + term2 * a * tilde_b * dt)
        numerator = term1 + jnp.sqrt(root_term)
        y1 = numerator / term2

        return y1

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
        del made_jump
        x0 = y0
        y0 = solver_state  # this is the square root of x0

        assert isinstance(y0, jax.Array)
        assert y0.shape == ()

        y_coarse = self._substep(terms, t0, t1, y0, args)
        x_coarse = jnp.square(y_coarse)
        if self.half_stepping:
            dt = t1 - t0
            t_half = t0 + dt / 2.0
            y_half = self._substep(terms, t0, t_half, y0, args)
            y1 = self._substep(terms, t_half, t1, y_half, args)
            x1 = jnp.square(y1)
            error = jnp.abs(x1 - x_coarse)
        else:
            x1 = x_coarse
            y1 = y_coarse
            error = None
        dense_info = dict(y0=x0, y1=x1)
        return x1, error, dense_info, y1, RESULTS.successful

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)
