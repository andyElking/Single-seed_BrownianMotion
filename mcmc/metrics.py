import math
from functools import partial, reduce
from operator import mul
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import ot
from jax import Array


def adjust_max_len(max_len, data_dim):
    if data_dim >= 4:
        exponent = math.ceil(math.log2(max_len / math.sqrt(data_dim)))
        max_len = 2 ** int(exponent)
    return max_len


def truncate_samples(x, max_len: Optional[int]):
    if max_len is None:
        return x

    data_dim = reduce(mul, x.shape[1:], 1)
    max_len = adjust_max_len(max_len, data_dim)
    if x.shape[0] > max_len:
        x = x[:max_len]
    return x


def compute_w2(x, y, num_iters, max_len: Optional[int] = 2**11):
    x = truncate_samples(x, max_len)
    y = truncate_samples(y, max_len)
    source_samples = np.array(x)
    target_samples = np.array(y)
    source_weights = np.ones(source_samples.shape[0]) / source_samples.shape[0]
    target_weights = np.ones(target_samples.shape[0]) / target_samples.shape[0]
    mm = ot.dist(source_samples, target_samples)
    return ot.emd2(source_weights, target_weights, mm, numItermax=num_iters)


@partial(jax.jit, static_argnames=("max_len_x", "max_len_y"))
def compute_energy(
    x: Array,
    y: Array,
    max_len_x: Optional[int] = 2**15,
    max_len_y: Optional[int] = 2**15,
):
    assert y.ndim == x.ndim
    x = truncate_samples(x, max_len_x)
    y = truncate_samples(y, max_len_y)

    @partial(jax.vmap, in_axes=(None, 0))
    def _dist_single(_x, _y_single):
        assert _x.ndim == _y_single.ndim + 1, f"{_x.ndim} != {_y_single.ndim + 1}"
        diff = _x - _y_single
        if x.ndim > 1:
            # take the norm over all axes except the first one
            diff = jnp.sqrt(jnp.sum(diff**2, axis=tuple(range(1, diff.ndim))))
        return jnp.mean(jnp.abs(diff))

    def dist(_x, _y):
        assert _x.ndim == _y.ndim
        return jnp.mean(_dist_single(_x, _y))

    return 2 * dist(x, y) - dist(x, x) - dist(y, y)
