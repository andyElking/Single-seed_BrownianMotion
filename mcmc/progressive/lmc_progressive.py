import time
from typing import Callable, Optional

import jax.random as jr

from ..lmc import run_simple_lmc_numpyro
from ..abstract_method import AbstractMethod


class ProgressiveLMC(AbstractMethod):
    def __init__(
        self,
        lmc_kwargs: dict,
        get_previous_result_filename: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(get_previous_result_filename)
        self.lmc_kwargs = lmc_kwargs
        self.method_name = str(lmc_kwargs["solver"].__class__.__name__)
        if ("pid" in lmc_kwargs) and (lmc_kwargs["pid"] is not None):
            self.method_name += "_ADAP"

    def run(self, key, model, model_args, result_dict, config):
        num_particles = config["num_particles"]
        start_time = time.time()
        samples, cumulative_evals, avg_accepted, avg_rejected = run_simple_lmc_numpyro(
            jr.key(0), model, model_args, num_particles, **self.lmc_kwargs
        )
        wall_time = time.time() - start_time

        aux_output = {"cumulative_evals": cumulative_evals, "wall_time": wall_time, "avg_accepted": avg_accepted, "avg_rejected": avg_rejected}
        return samples, aux_output
