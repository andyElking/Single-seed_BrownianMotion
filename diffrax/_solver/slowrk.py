from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import AbstractSRK, GeneralNoiseCoefficients, StochasticButcherTableau


cfs_w = GeneralNoiseCoefficients(
    a=(
        np.array([0.0]),
        np.array([0.0, 0.5]),
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.75, 0.0]),
        np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    ),
    b=np.array([0.0, 1 / 6, 1 / 3, 1 / 3, 1 / 6, 0.0, 0.0]),
)

cfs_hh = GeneralNoiseCoefficients(
    a=(
        np.array([0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.5, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ),
    b=np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0]),
)

_tab = StochasticButcherTableau(
    c=np.array([0.5, 0.5, 0.5, 0.5, 0.75, 1.0]),
    b_sol=np.array([1 / 3, 0.0, 0.0, 0.0, 0.0, 2 / 3, 0.0]),
    b_error=None,
    a=[
        np.array([0.5]),
        np.array([0.5, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0]),
        np.array([0.75, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    cfs_w=cfs_w,
    cfs_hh=cfs_hh,
    additive_noise=False,
    ignore_stage_f=np.array([False, True, True, True, True, False, True]),
    ignore_stage_g=np.array([True, False, False, False, False, True, False]),
)


class SlowRK(AbstractSRK, AbstractStratonovichSolver):
    r"""SLOW-RK method for SDEs by James Foster.
    Applied to SDEs with commutative noise, it converges strongly with order 1.5.
    Can be used for SDEs with non-commutative noise, but then it only converges
    strongly with order 0.5.

    """

    tableau: ClassVar[StochasticButcherTableau] = _tab

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5
