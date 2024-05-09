from typing import Callable, Optional, TypeAlias

import jax.numpy as jnp
from jax.nn import relu
from jaxtyping import PyTree

from .. import AbstractTerm, RESULTS
from .._custom_types import Args, BoolScalarLike, IntScalarLike, RealScalarLike, VF, Y
from .base import AbstractStepSizeController


_ControllerState: TypeAlias = None


class CIRController(AbstractStepSizeController[None, Optional[RealScalarLike]]):
    """Step size controller for the CIR process."""

    ctol: RealScalarLike
    dtmax: RealScalarLike
    dtmin: RealScalarLike

    def wrap(self, direction: IntScalarLike) -> "AbstractStepSizeController":
        return self

    def desired_step_size(self, y_min):
        y_min = relu(y_min)
        step_size = jnp.power(y_min * self.ctol, 2.0 / 3.0)
        return jnp.clip(step_size, self.dtmin, self.dtmax)

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
        step_size = self.desired_step_size(y0)
        return t0 + step_size, None

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
        del args, y_error, error_order, controller_state
        y_min = jnp.minimum(y0, y1_candidate)
        desired = self.desired_step_size(y_min)

        accept = t1 - t0 < 1.1 * desired
        new_t0 = jnp.where(accept, t1, t0)
        new_dt = jnp.where(accept, self.desired_step_size(y1_candidate), desired)
        new_dt = jnp.clip(new_dt, self.dtmin, self.dtmax)
        new_t1 = new_t0 + new_dt

        return accept, new_t0, new_t1, False, None, RESULTS.successful
