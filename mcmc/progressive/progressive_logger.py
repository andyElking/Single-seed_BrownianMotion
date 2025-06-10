import jax.numpy as jnp

from ..logger import AbstractLogger


class ProgressiveLogger(AbstractLogger):
    def make_log_string(self, method_name: str, method_dict: dict) -> str:
        best_acc = jnp.max(method_dict["test_acc"])
        if "test_acc_best80" in method_dict:
            best_acc80 = jnp.max(method_dict["test_acc_best80"])
            acc_top_str = f" acc top 80%: {best_acc80:.4},"
        elif "test_acc_best90" in method_dict:
            best_acc90 = jnp.max(method_dict["test_acc_best90"])
            acc_top_str = f" acc top 90%: {best_acc90:.4},"
        else:
            acc_top_str = ""

        best_energy = jnp.min(method_dict["energy_err"])
        best_w2 = jnp.min(method_dict["w2"])
        str_out = (
            f"{method_name}: acc: {best_acc:.4},{acc_top_str}"
            f" energy: {best_energy:.3e}, w2: {best_w2:.3e}"
        )
        if ("avg_accepted" in method_dict) and (method_dict["avg_accepted"] is not None):
            str_out += f", avg accepted: {method_dict['avg_accepted']:.3f}"
        if ("avg_rejected" in method_dict) and (method_dict["avg_rejected"] is not None):
            str_out += f", avg rejected: {method_dict['avg_rejected']:.3f}"
        if ("wall_time" in method_dict) and (method_dict["wall_time"] is not None):
            str_out += f", wall time: {method_dict['wall_time']:.3f}s"
        if ("cumulative_evals" in method_dict) and (method_dict["cumulative_evals"] is not None):
            str_out += f", cumulative evals: {int(method_dict['cumulative_evals'][-1])}"
        return str_out
