import pickle
from typing import Callable, Optional

import jax.random as jr

from .abstract_method import AbstractMethod
from .logger import AbstractLogger
from .progressive import AbstractProgressiveEvaluator


def run_experiment(
    key,
    model,
    model_args,
    model_name,
    methods: list[AbstractMethod],
    config,
    evaluator: AbstractProgressiveEvaluator,
    logger: AbstractLogger,
    get_gt_fun: Callable,
    gt_eval_fun: Optional[Callable],
    get_result_filename: Optional[Callable[[str], str]],
):
    key_gt, key = jr.split(key, 2)
    gt = get_gt_fun(model, model_name, model_args, config, key_gt)
    if gt_eval_fun is not None:
        gt_str = gt_eval_fun(gt, config) + "\n"
        logger.print_log(gt_str)

    result_dict = {"model_name": model_name}

    for method in methods:
        loaded_dict = method.previous_results(model_name)
        if loaded_dict is None:
            key, key_sample, key_eval = jr.split(key, 3)
            samples, aux_output = method.run(
                key_sample, model, model_args, result_dict, config
            )
            method_dict = evaluator.eval(
                samples, aux_output, gt, config, None, key_eval
            )
            del samples, aux_output
        else:
            method_dict = loaded_dict
        result_dict[method.method_name] = method_dict
        # print("method dict:")
        # print(method_dict)
        logger.log_method(method.method_name, method_dict)

    if get_result_filename is not None:
        result_filename = get_result_filename(model_name)
        with open(result_filename, "wb") as f:
            pickle.dump(result_dict, f)
