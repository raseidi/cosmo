import os
import pandas as pd
import torch
from functools import wraps
import pm4py


PATH_ROOT = "data/"


def clear_cache(log=None):
    if log:
        pass
    # remove every "cached_log.pkl" file
    for root, dirs, files in os.walk(PATH_ROOT):
        for file in files:
            if file == "cached_log.pkl":
                os.remove(os.path.join(root, file))


def read_log(path):
    """Read log from path, either .pkl, .csv, or .xes.

    The method ensures that the log is cached as a .pkl file."""
    path = os.path.join(path, "log.pkl")
    if os.path.exists(path):
        log = pd.read_pickle(path)
        return log
    elif os.path.exists(path.replace(".pkl", ".csv")):
        log = pd.read_csv(path.replace(".pkl", ".csv"))
    elif os.path.exists(path.replace(".pkl", ".xes")):
        log = pm4py.read_xes(path.replace(".pkl", ".xes"))
    else:
        raise ValueError("Log not found at path: {}".format(path))

    return log


# def cache(file_name="cached_log", format=".pkl", fn_name=None):
def cache(fn):
    """Cache the log dataframe

    Check if a preprocessed log dataframe exists in the cache directory.
    If it does, load it. If it doesn't, preprocess the log dataframe and save it to the cache directory.

    Args:
        fn (function): function that returns a log dataframe

    Returns:
        function: decorator wrapper function that caches the log dataframe
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        cache_path = os.path.join("data", f"{fn.__name__}", "cached_log.pkl")
        if os.path.exists(cache_path):
            log = pd.read_pickle(cache_path)
        else:
            log = fn(*args, **kwargs)
            log.to_pickle(cache_path)

        return log.copy()

    return wrapper


def ensure_dir(dir_name: str):
    """Creates folder if it does not exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def collate_fn(batch: list[dict]):
    """Helper function to pad multiple sequences in batch

    Args:
        batch (list): a list of cases.
        Each case is a dictionary of event features and tensors.

    Returns:
        formated_batch (dict): a dictionary of event features and padded tensors.
        The event features are grouped by their prefix. For example,
        if the features are "cat_activity", "cat_resource", "num_time_since_last_event",
        "num_time_accumulated", then the formated_batch will be
        {
            "cat": tensor([batch_size, max_seq_len, num_cat_features]),
            "num": tensor([batch_size, max_seq_len, num_num_features]),
        }
    """
    values = {}
    for dict_b in batch:
        for feature, tensors in dict_b.items():
            if feature not in values:
                values[feature] = []

            # if ndim == 1 -> sequence
            if tensors.ndim == 1 and feature != "target":
                values[feature].append(tensors.unsqueeze(-1))
            # else: constraint array of shape (1, n_constraints)
            elif feature == "target":
                values[feature].append(tensors)
            else:
                values[feature].append(tensors)

    # pad values
    for key, value in values.items():
        feature_dtype = value[0].dtype
        padding_value = 0 if feature_dtype == torch.long else 0.0
        values[key] = torch.nn.utils.rnn.pad_sequence(
            [f for f in value], batch_first=True, padding_value=padding_value
        )

    # concat features of same prefix
    formated_batch = {}
    for key, value in values.items():
        prefix = key.split("_")[0]
        if prefix not in formated_batch:
            formated_batch[prefix] = value
        else:
            formated_batch[prefix] = torch.cat([formated_batch[prefix], value], dim=-1)
    return formated_batch
