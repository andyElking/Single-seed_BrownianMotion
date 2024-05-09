from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import AbstractSRK, AdditiveNoiseCoefficients, StochasticButcherTableau


cfs_w = AdditiveNoiseCoefficients(
    a=np.array([0.5]),
    b=np.array(1.0),
)

cfs_hh = AdditiveNoiseCoefficients(
    a=np.array([1.0]),
    b=np.array(0.0),
)

_tab = StochasticButcherTableau(
    c=np.array([]),
    b_sol=np.array([1.0]),
    b_error=None,
    a=[],
    cfs_w=cfs_w,
    cfs_hh=cfs_hh,
)


class SEA(AbstractSRK, AbstractStratonovichSolver):
    r"""Shifted Euler method for SDEs with additive noise.
     It has a local error of $O(h^2)$ compared to
     standard Euler-Maruyama, which has $O(h^{1.5})$.
     Uses one evaluation of the vector field per step and
     has order 1 for additive noise SDEs.

    Based on equation $(5.8)$ in
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

    tableau: ClassVar[StochasticButcherTableau] = _tab

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 1
