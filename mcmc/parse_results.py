import glob
import math
import os
import pickle

import numpy as np

def result_dict_to_string(result_dict):
    result_str = ""
    ess_per_sample = result_dict["ess_per_sample"]
    result_str += f"\nESS per sample: {ess_per_sample:.4}"
    evals_per_sample = result_dict["grad_evals_per_sample"]
    if evals_per_sample is not None:
        avg_evals = np.mean(evals_per_sample)
        result_str += f", grad evals per sample: {avg_evals:.4}"
        # grad evals per effective sample
        gepes = avg_evals / ess_per_sample
        result_str += f", GEPS/ESS: {gepes:.4}"

    energy_gt = result_dict["energy_gt"]
    if energy_gt is not None:
        result_str += f"\nEnergy dist vs ground truth: {energy_gt:.4}"

    w2 = result_dict["w2"]
    if w2 is not None:
        result_str += f", Wasserstein-2 error: {w2:.4}"

    test_acc = result_dict["test_accuracy"]
    if test_acc is not None:
        test_acc_best80 = result_dict["top90_accuracy"]
        result_str += (
            f"\nTest_accuracy: {test_acc:.4}, top 80% accuracy: {test_acc_best80:.4}"
        )

    return result_str
