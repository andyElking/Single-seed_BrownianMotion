import time
from typing import Callable, Optional

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model, Predictive

from ..abstract_method import AbstractMethod


class ProgressiveNUTS(AbstractMethod):
    def __init__(
        self,
        num_warmup: int,
        chain_len: int,
        get_previous_result_filename: Optional[Callable[[str], str]] = None,
        prior_start: bool = False,
    ):
        super().__init__(get_previous_result_filename)
        self.method_name = "NUTS"
        self.num_warmup = num_warmup
        self.chain_len = chain_len
        self.prior_start = prior_start

    def run(self, key, model, model_args, result_dict, config):
        num_particles = config["num_particles"]

        key_init, key_warmup, key_run = jr.split(key, 3)
        if self.prior_start:
            x0 = Predictive(model, num_samples=num_particles)(key, *model_args)
            x0.pop("obs", None)
            x0.pop("Y", None)
        else:
            model_info = initialize_model(key, model, model_args=model_args)
            x0 = model_info.param_info.z
            x0 = jtu.tree_map(lambda x: jnp.tile(x, (num_particles, 1)), x0)

        # run NUTS and record wall time
        start_nuts = time.time()
        nuts = MCMC(
            NUTS(model),
            num_warmup=self.num_warmup,
            num_samples=self.chain_len - self.num_warmup,
            num_chains=num_particles,
            chain_method="vectorized",
        )
        nuts.warmup(
            key_warmup,
            *model_args,
            init_params=x0,
            extra_fields=("num_steps",),
            collect_warmup=True,
        )
        warmup_steps = jnp.reshape(
            nuts.get_extra_fields()["num_steps"], (num_particles, -1)
        )
        warmup_samples = nuts.get_samples(group_by_chain=True)
        nuts.run(key_run, *model_args, extra_fields=("num_steps",))
        run_samples = nuts.get_samples(group_by_chain=True)
        time_nuts = time.time() - start_nuts
        samples = jtu.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=1), warmup_samples, run_samples
        )
        run_steps = jnp.reshape(
            nuts.get_extra_fields()["num_steps"], (num_particles, -1)
        )
        steps_nuts = jnp.concatenate((warmup_steps, run_steps), axis=-1)
        steps_nuts = jnp.mean(steps_nuts, axis=0)
        cumulative_evals = jnp.cumsum(steps_nuts)

        aux_output = {"cumulative_evals": cumulative_evals, "wall_time": time_nuts, "avg_accepted": jnp.sum(steps_nuts), "avg_rejected": None}
        return samples, aux_output
