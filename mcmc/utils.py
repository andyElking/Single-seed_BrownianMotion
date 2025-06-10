import jax
import jax.random as jr
import jax.tree_util as jtu
from numpyro import handlers


def get_prior_samples(key, model, model_args, num_samples):
    model_args_short = jtu.tree_map(lambda x: x[:3], model_args)

    def prior_from_key(key):
        seeded_model = handlers.seed(model, rng_seed=key)
        return seeded_model(*model_args_short)

    return jax.vmap(prior_from_key)(jr.split(key, num_samples))
