import numpy as np
from jaxtyping import PyTree

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import scipy
from numpyro.infer import MCMC, NUTS  # noqa: F401
from numpyro.infer.util import initialize_model, Predictive  # noqa: F401
from .helpers import make_underdamped_langevin_term, SDE
from .lmc import get_x0
from .logreg_utils import get_model_and_data


def get_logreg_sde(data_name, t1, key, dtype=jnp.float32, x0=None):
    dataset = scipy.io.loadmat("../mcmc/mcmc_data/benchmarks.mat")
    model, model_args, test_args = get_model_and_data(dataset, data_name)

    def cast_to_type(x):
        if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
            return jnp.asarray(x, dtype=dtype)
        else:
            return x

    model = jtu.tree_map(cast_to_type, model)
    model_args = jtu.tree_map(cast_to_type, model_args)
    model_info = initialize_model(key, model, model_args=model_args)
    log_p = jax.jit(model_info.potential_fn)
    num_particles = 1

    if x0 is None:
        x0 = get_x0(model, model_args, num_particles, key, True)

    x0 = jtu.tree_map(lambda x: jnp.astype(x, dtype), x0)

    grad_log_p = jax.jit(jax.grad(log_p))

    def grad_f(x, _):
        return grad_log_p(x)

    v0 = jtu.tree_map(lambda x: jnp.zeros_like(x), x0)
    y0 = (x0, v0)

    def get_shape(x):
        shape = jnp.shape(x)
        assert shape[0] == num_particles
        return jax.ShapeDtypeStruct(shape[1:], x.dtype)

    w_shape: PyTree[jax.ShapeDtypeStruct] = jtu.tree_map(get_shape, x0)

    gamma, u = 1.0, 1.0

    def get_terms(bm):
        return make_underdamped_langevin_term(gamma, u, grad_f, bm)

    return SDE(get_terms, None, y0, 0.0, t1, w_shape)