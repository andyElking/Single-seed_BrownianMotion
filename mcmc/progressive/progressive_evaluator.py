from abc import abstractmethod
from functools import partial
from typing import Optional

import jax
import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp
from jax import Array

from ..logreg_utils import test_accuracy, vec_dict_to_array
from ..metrics import compute_energy, compute_w2


def compute_metrics(sample_slice, ground_truth, x_test, labels_test):
    if x_test is not None and labels_test is not None:
        test_acc, test_acc_best80 = test_accuracy(x_test, labels_test, sample_slice)
    else:
        test_acc, test_acc_best80 = None, None

    return {
        "test_acc": test_acc,
        "test_acc_best80": test_acc_best80,
    }


class AbstractProgressiveEvaluator:
    def __init__(self, num_points=32):
        self.num_points = num_points

    @abstractmethod
    def vectorisable_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict[str, Optional[Array]]:
        raise NotImplementedError

    @abstractmethod
    def sequential_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def preprocess_samples(self, samples, config):
        raise NotImplementedError

    def eval(self, samples, aux_output, ground_truth, config, model, rng):
        cumulative_evals = aux_output["cumulative_evals"]
        wall_time = aux_output["wall_time"]
        avg_accepted = aux_output.get("avg_accepted", 0.0)
        avg_rejected = aux_output.get("avg_rejected", 0.0)
        samples = self.preprocess_samples(samples, config)

        num_chains, chain_len = jtu.tree_leaves(samples)[0].shape[:2]

        assert jnp.shape(cumulative_evals) == (
            chain_len,
        ), f"{cumulative_evals.shape} != {(chain_len,)}"

        assert chain_len >= self.num_points and chain_len % self.num_points == 0
        eval_interval = chain_len // self.num_points
        cumulative_evals = cumulative_evals[::eval_interval]
        samples = jtu.tree_map(lambda x: x[:, ::eval_interval], samples)
        assert jtu.tree_all(
            jtu.tree_map(
                lambda x: x.shape[:2] == (num_chains, self.num_points), samples
            )
        ), (
            f"expected shapes prefixed by {(num_chains, self.num_points)}"
            f" but got {jtu.tree_map(lambda x: x.shape, samples)}"
        )
        assert jnp.shape(cumulative_evals) == (self.num_points,)

        # now we go along chain_len and compute the metrics for each step
        partial_metrics = lambda _s, _k: self.vectorisable_metrics(
            _s, ground_truth, config, model, _k
        )
        # vectorize over the chain_len dimension
        rngs = jr.split(rng, self.num_points)
        vec_metrics = jax.jit(jax.vmap(partial_metrics, in_axes=(1, 0)))
        vec_dict = vec_metrics(samples, rngs)

        def get_slice(samples, i):
            return jtu.tree_map(lambda x: x[:, i], samples)

        # compute metrics which cannot be vectorised (like W2)
        seq_dict: dict[str, list] = {}
        for i in range(self.num_points):
            seq_out = self.sequential_metrics(
                get_slice(samples, i), ground_truth, config, model, rngs[i]
            )
            for key, value in seq_out.items():
                if key not in seq_dict:
                    seq_dict[key] = []
                seq_dict[key].append(value)

        result_dict = {
            "cumulative_evals": cumulative_evals,
            "wall_time": wall_time,
            "avg_accepted": avg_accepted,
            "avg_rejected": avg_rejected,
        }
        for key, value in vec_dict.items():
            assert jnp.shape(value) == (self.num_points,), f"{key} has shape {value.shape}, expected {(self.num_points,)}"
            result_dict[key] = value
        for key, value in seq_dict.items():
            result_dict[key] = jnp.array(value)
            assert jnp.shape(value) == (self.num_points,), f"{key} has shape {jnp.shape(value)}, expected {(self.num_points,)}"

        return result_dict


class ProgressiveEvaluator(AbstractProgressiveEvaluator):
    def __init__(self, num_iters_w2=100000, max_samples_w2=2**11, num_points=32):
        self.num_iters_w2 = num_iters_w2
        self.max_samples_w2 = max_samples_w2
        super().__init__(num_points=num_points)

    def vectorisable_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict[str, Optional[Array]]:

        # We use bootstrapping to compute the metrics and the standard errors
        n_samples = sample_slice.shape[0]

        def one_bootstrap(key):
            # Sample indices with replacement.
            idx = jr.randint(key, shape=(n_samples,), minval=0, maxval=n_samples)
            boot_samples = sample_slice[idx]
            return compute_metrics(
                boot_samples, ground_truth, *config["test_args"]
            )

        bootstrap_reps = config.get("bootstrap_reps", 10)
        keys = jr.split(key, bootstrap_reps)

        results_boot = jax.jit(jax.vmap(one_bootstrap, in_axes=0))(keys)
        # Compute the mean and standard error for each metric
        result = {}
        for k, v in results_boot.items():
            mean = jnp.mean(v, axis=0)
            se = jnp.std(v, axis=0, ddof=1)
            result[k] = mean
            result[f"{k}_se"] = se

        result = jtu.tree_map(jnp.squeeze, result)
        for k, v in result.items():
            assert v.shape == ()

        return result

    def sequential_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict[str, Optional[Array]]:
        # We use bootstrapping to compute the metrics and the standard errors
        n_samples = sample_slice.shape[0]

        def one_bootstrap(key):
            # Sample indices with replacement.
            idx = jr.randint(key, shape=(n_samples,), minval=0, maxval=n_samples)
            boot_samples = sample_slice[idx]
            energy_err = compute_energy(
                boot_samples, ground_truth, max_len_x=2 ** 15, max_len_y=2 ** 15
            )

            if self.num_iters_w2 > 0:
                w2 = compute_w2(
                    boot_samples,
                    ground_truth,
                    self.num_iters_w2,
                    self.max_samples_w2,
                )
            else:
                w2 = None
            return energy_err, w2

        bootstrap_reps = config.get("bootstrap_reps", 10)
        keys = jr.split(key, bootstrap_reps)
        energy_errs = []
        w2s = []
        for i, k in enumerate(keys):
            energy_err, w2 = one_bootstrap(k)
            energy_errs.append(energy_err)
            if w2 is not None:
                w2s.append(w2)
        energy_err = jnp.mean(jnp.array(energy_errs), axis=0)
        energy_err_se = jnp.std(jnp.array(energy_errs), axis=0, ddof=1)

        result = {
            "energy_err": energy_err,
            "energy_err_se": energy_err_se,
        }

        if len(w2s) > 0:
            w2 = jnp.mean(jnp.array(w2s), axis=0)
            w2_se = jnp.std(jnp.array(w2s), axis=0, ddof=1)
            result["w2"] = w2
            result["w2_se"] = w2_se

        result = jtu.tree_map(jnp.squeeze, result)
        for k, v in result.items():
            assert v.shape == ()

        return result

    def preprocess_samples(self, samples, config):
        if isinstance(samples, dict):
            samples = vec_dict_to_array(samples)
        return samples
