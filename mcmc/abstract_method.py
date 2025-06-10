import glob
import os
import pickle
from typing import Callable, Optional

from jax import Array


class AbstractMethod:
    get_previous_result_filename: Optional[Callable[[str], str]] = None
    method_name: str

    def __init__(
        self, get_previous_result_filename: Optional[Callable[[str], str]] = None
    ):
        self.get_previous_result_filename = get_previous_result_filename

    def previous_results(self, model_name: str) -> Optional[dict]:
        if self.get_previous_result_filename is None:
            return None
        prev_results_filename = self.get_previous_result_filename(model_name)

        # the filename could be generic, in which case we return the latest
        # result that matches the filename
        filenames = glob.glob(prev_results_filename)
        filenames.sort(key=os.path.getmtime)
        latest_filename = filenames[-1]
        with open(latest_filename, "rb") as f:
            prev_results = pickle.load(f)
        return prev_results[self.method_name]

    def run(self, key, model, model_args, result_dict, config) -> tuple[Array, dict]:
        """
        **Arguments**:

            - key: PRNGKey
            - model: a numpyro model to be sampled from
            - model_args: arguments to be passed to the model
            - result_dict: dict containing results from previous methods;
            this is used e.g. for LMC to use a similar number of evaluations as NUTS
            - config: additional data needed for the method

        **Returns**:

            - samples: Array, shape (num_chains, num_samples, d)
            - aux_output: dict to be fed into the evaluation function
        """
        raise NotImplementedError
