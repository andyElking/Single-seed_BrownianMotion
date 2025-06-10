import os

import jax
import numpy as np
import numpyro
from jax import numpy as jnp, random as jr, tree_util as jtu
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

from .metrics import compute_energy


def get_model(data_dim):
    def model(x, labels):
        x_var = jnp.var(x, axis=0)
        W = numpyro.sample(
            "W",
            dist.Normal(jnp.zeros(data_dim), 0.5 / x_var),  # pyright: ignore
        )
        b = numpyro.sample("b", dist.Normal(jnp.zeros((1,)), 1))  # pyright: ignore
        logits = jnp.sum(W * x, axis=-1) + b
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    return model


def train_test_split(x, labels, n_train_max):
    n_train = min(int(0.8 * x.shape[0]), n_train_max)
    n_test = min(x.shape[0] - n_train, n_train_max)
    x_train = x[:n_train]
    labels_train = labels[:n_train]
    x_test = x[n_train : n_train + n_test]
    labels_test = labels[n_train : n_train + n_test]
    return x_train, labels_train, x_test, labels_test


def get_model_and_data(data, name):
    if name in ["tbp", "isolet", "isolet_ab"]:
        return get_uci_data(name)

    dset = data[name][0, 0]
    x = dset["x"]
    labels = jnp.squeeze(dset["t"])
    # labels are -1 and 1, convert to 0 and 1
    labels = (labels + 1) / 2
    n, data_dim = x.shape
    print(f"Data shape: {x.shape}")

    # randomly shuffle the data
    perm = jax.random.permutation(jr.PRNGKey(0), n)
    x = x[perm]
    labels = labels[perm]

    x_train, labels_train, x_test, labels_test = train_test_split(x, labels, 700)
    print(
        f"x_train shape: {x_train.shape}, labels_train shape: {labels_train.shape}"
        f"x_train dtype: {x_train.dtype}, labels_train dtype: {labels_train.dtype}"
    )

    return get_model(data_dim), (x_train, labels_train), (x_test, labels_test)


def get_uci_data(name):
    x = np.load(f"mcmc_data/{name}_x.npy")
    labels = np.load(f"mcmc_data/{name}_y.npy")
    labels = jnp.array(labels, dtype=jnp.float32)
    x = jnp.array(x, dtype=jnp.float32)
    data_dim = x.shape[1]
    x_train, labels_train, x_test, labels_test = train_test_split(x, labels, 700)
    print(
        f"x_train shape: {x_train.shape}, labels_train shape: {labels_train.shape},"
        f" x_test shape: {x_test.shape}, labels_test shape: {labels_test.shape}"
    )
    return get_model(data_dim), (x_train, labels_train), (x_test, labels_test)


def get_gt_logreg(model, model_name, model_args, config, key):
    filename = f"ground_truth/{model_name}_ground_truth.npy"
    # if ground_truth is not computed, compute it
    if not os.path.exists(filename):
        num_chains = 2**8
        key_run, key_perm = jr.split(key, 2)
        gt_nuts = MCMC(
            NUTS(model),
            num_warmup=2**9,
            num_samples=2**9,
            num_chains=num_chains,
            chain_method="vectorized",
        )
        gt_nuts.run(jr.PRNGKey(0), *model_args)
        gt = vec_dict_to_array(gt_nuts.get_samples())
        # shuffle the ground truth samples
        permute = jax.jit(lambda x: jr.permutation(key_perm, x, axis=0))
        gt = jtu.tree_map(permute, gt)
        np.save(filename, gt)
    else:
        gt = np.load(filename)
    return gt


def dict_to_array(dct: dict):
    b = dct["b"]
    lst = [b, dct["W"]]
    if "alpha" in dct:
        alpha = dct["alpha"]
        alpha = jnp.expand_dims(alpha, alpha.ndim)
        lst = [alpha] + lst
    return jnp.concatenate(lst, axis=-1)


vec_dict_to_array = jax.jit(jax.vmap(dict_to_array, in_axes=0, out_axes=0))


def eval_gt_logreg(gt, config):
    x_test, labels_test = config["test_args"]
    size_gt_half = int(gt.shape[0] // 2)
    gt_energy_bias = compute_energy(gt[:size_gt_half], gt[size_gt_half:])
    gt_test_acc, gt_test_acc_best80 = test_accuracy(x_test, labels_test, gt)
    str_gt = (
        f"GT energy bias: {gt_energy_bias:.3e}, test acc: {gt_test_acc:.4},"
        f" test acc top 80%: {gt_test_acc_best80:.4}"
    )
    return str_gt


def predict(x, samples):
    b = samples[:, 0]
    w = samples[:, 1:]
    logits = jnp.sum(w * x, axis=-1) + b
    # apply sigmoid
    return 1.0 / (1.0 + jnp.exp(-logits))


def test_accuracy(x_test, labels_test, samples):
    if isinstance(samples, dict):
        samples = vec_dict_to_array(samples)
    assert x_test.shape[1] + 1 == samples.shape[-1], (
        f"The last dim of {x_test.shape} should be the"
        f" last dim of {samples.shape} minus 1"
    )
    sample_dim = samples.shape[-1]
    samples = jnp.reshape(samples, (-1, sample_dim))
    if samples.shape[0] > 2**10:
        samples = samples[: 2**10]

    func = jax.jit(jax.vmap(lambda x: predict(x, samples), in_axes=0, out_axes=0))
    predictions = func(x_test)
    assert predictions.shape == (
        labels_test.shape[0],
        samples.shape[0],
    ), f"{predictions.shape} != {(labels_test.shape[0], samples.shape[0])}"

    labels_test = jnp.reshape(labels_test, (labels_test.shape[0], 1))
    is_correct = jnp.abs(predictions - labels_test) < 0.5
    accuracy_per_sample = jnp.mean(is_correct, axis=0)

    avg_accuracy = jnp.mean(accuracy_per_sample)

    len20 = int(0.2 * accuracy_per_sample.shape[0])
    best_sorted = jnp.sort(accuracy_per_sample)[len20:]
    accuracy_best80 = jnp.mean(best_sorted)
    return avg_accuracy, accuracy_best80
