from typing import Callable, Optional, TypeAlias

import jax.numpy as jnp
from jaxtyping import PyTree

from .. import AbstractTerm, RESULTS
from .._custom_types import Args, BoolScalarLike, IntScalarLike, RealScalarLike, VF, Y
from .base import AbstractStepSizeController


_ControllerState: TypeAlias = None


class SDEController(AbstractStepSizeController[None, Optional[RealScalarLike]]):
    """Step size controller for the CIR process."""

    ctol: RealScalarLike
    dtmax: RealScalarLike
    dtmin: RealScalarLike

    def wrap(self, direction: IntScalarLike) -> "AbstractStepSizeController":
        return self

    def desired_step_size(self, y, err, dt: RealScalarLike):
        tol = self.ctol * (0.001 + jnp.abs(y))
        step = jnp.where(tol * dt < jnp.square(err), dt / 2, 2 * dt)
        return jnp.clip(step, self.dtmin, self.dtmax)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: Optional[RealScalarLike],
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, None]:
        del terms, t1, dt0, args, func, error_order
        return t0 + self.dtmax, None

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: _ControllerState,
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _ControllerState,
        RESULTS,
    ]:
        del args, error_order, controller_state
        dt = t1 - t0
        y_min = jnp.minimum(y0, y1_candidate)
        new_dt = self.desired_step_size(y_min, y_error, dt)

        accept = dt < 1.1 * new_dt
        new_t0 = jnp.where(accept, t1, t0)
        new_t1 = new_t0 + new_dt

        return accept, new_t0, new_t1, False, None, RESULTS.successful
