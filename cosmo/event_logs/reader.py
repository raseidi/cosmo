import os
import pm4py
import pandas as pd
from cosmo.event_logs.utils import cache, read_log
from cosmo.event_logs.preprocess import common_preprocessing

""" 
Here we process the logs stored in the 'data' folder. Each log is organized
within its own folder, containing the log file in either PKL or XES format,
along with train/test split and declare constraints in CSV format. The 
folder structure is consistent for all logs, with only the log folder name
varying.

The goal is to read the train/test split, if available, and return a single
log along with the declare constraints, which can be extracted using the
`extract_declare.py` script. If the split is not available, i.e., for non
bpi logs, then we use pm4py to split the data.

NOTE: cache must be the outermost decorator, otherwise it will cache only 
the decorated function and ignore the other decorators

Basic structure of paths:

data/
    log_name/
        declare/
            constraints.csv
        train_test/
            train.csv
            test.csv
        log.csv
        log.xes
    log_name2/
        ...
"""

PATH_ROOT = "data/"
AVAILABLE_LOGS = [
    "bpi12",
    "bpi13_incidents",
    "bpi13_problems",
    "bpi15",
    "bpi17",
    "bpi19",
    "bpi20_domestic",
    "bpi20_international",
    "bpi20_permit",
    "bpi20_prepaid",
    "bpi20_req4pay",
    "sepsis",
]

DECLARE_TEMPLATES = {
    "existence": [
        "Existence1",
        "Absence1",
        "Exactly1",
        "End",
    ],
    "choice": [
        "Choice",
        "Exclusive Choice",
    ],
    "positive relations": [
        "Alternate Precedence",
        "Alternate Response",
        "Chain Precedence",
        "Chain Response",
        "Precedence",
        "Responded Existence",
        "Response",
    ],
    "negative relations": [
        "Not Chain Precedence",
        "Not Chain Response",
        "Not Precedence",
        "Not Responded Existence",
        "Not Response",
    ],
}


def get_declare(dataset, templates="all"):
    if templates != "all":
        templates = DECLARE_TEMPLATES.get(templates, None)
        if templates is None:
            raise ValueError(
                f"Invalid template. Choose from {list(DECLARE_TEMPLATES.keys())} or 'all' to use all templates."
            )

    assert dataset in AVAILABLE_LOGS, f"Dataset {dataset} not available"

    declare_path = os.path.join(PATH_ROOT, dataset, "declare", "constraints.pkl")
    d = pd.read_pickle(declare_path)
    d.case_id = d.case_id.astype(str)
    d.set_index(["case_id"], inplace=True)

    # selecting templates
    # TODO: this version only works for single template or all templates
    if templates != "all":
        cols = [c for c in d.columns if c.split("[")[0] in templates]
        d = d[cols]

    # dropping constraints with variance less than 5% to improve training and avoid sparse matrix full of zeros
    constrs = d.var() > 0.05
    constrs = constrs[constrs].index.values
    d = d[constrs].reset_index()

    return d


@cache
@common_preprocessing
def bpi12():
    train = pd.read_csv(os.path.join(PATH_ROOT, "bpi12", "train_test", "train.csv"))
    test = pd.read_csv(os.path.join(PATH_ROOT, "bpi12", "train_test", "test.csv"))
    train["split"] = "train"
    test["split"] = "test"
    log = pd.concat([train, test])
    # drop duplicate columns

    return log


@cache
@common_preprocessing
def bpi13_problems():
    train = pd.read_csv(
        os.path.join(PATH_ROOT, "bpi13_problems", "train_test", "train.csv")
    )
    test = pd.read_csv(
        os.path.join(PATH_ROOT, "bpi13_problems", "train_test", "test.csv")
    )
    train["split"] = "train"
    test["split"] = "test"
    log = pd.concat([train, test])
    log = log.loc[:, ["case:concept:name", "activity", "time:timestamp", "split"]]

    return log


@cache
@common_preprocessing
def bpi13_incidents(path=None):
    # train = pd.read_csv(os.path.join(PATH_ROOT, "bpi13_incidents", "train_test", "train.csv"))
    # test = pd.read_csv(os.path.join(PATH_ROOT, "bpi13_incidents", "train_test", "test.csv"))

    # train["split"] = "train"
    # test["split"] = "test"
    # log = pd.concat([train, test])
    # log = log.loc[:, ["case:concept:name", "activity", "time:timestamp", "split"]]
    path = PATH_ROOT if path is None else path
    log_path = os.path.join(path, "bpi13_incidents")
    log = read_log(log_path)

    train_set, test_set = pm4py.split_train_test(log.copy())
    train_set.loc[:, "split"] = "train"
    test_set.loc[:, "split"] = "test"
    log = pd.concat([train_set, test_set])
    return log


@cache
@common_preprocessing
def bpi17():
    train = pd.read_csv(os.path.join(PATH_ROOT, "bpi17", "train_test", "train.csv"))
    test = pd.read_csv(os.path.join(PATH_ROOT, "bpi17", "train_test", "test.csv"))
    train["split"] = "train"
    test["split"] = "test"
    log = pd.concat([train, test])

    log = log.loc[:, ["case:concept:name", "concept:name", "time:timestamp", "split"]]
    return log


@cache
@common_preprocessing
def bpi19():
    train = pd.read_csv(os.path.join(PATH_ROOT, "bpi19", "train_test", "train.csv"))
    test = pd.read_csv(os.path.join(PATH_ROOT, "bpi19", "train_test", "test.csv"))
    train["split"] = "train"
    test["split"] = "test"
    log = pd.concat([train, test])
    return log


@cache
@common_preprocessing
def bpi20_req4pay():
    train = pd.read_csv(
        os.path.join(PATH_ROOT, "bpi20_req4pay", "train_test", "train.csv")
    )
    test = pd.read_csv(
        os.path.join(PATH_ROOT, "bpi20_req4pay", "train_test", "test.csv")
    )
    train["split"] = "train"
    test["split"] = "test"
    log = pd.concat([train, test])
    return log


# @cache
# @common_preprocessing
# def bpi20_domestic():
#     train = pd.read_csv(
#         os.path.join(PATH_ROOT, "bpi20_domestic", "train_test", "train.csv")
#     )
#     test = pd.read_csv(
#         os.path.join(PATH_ROOT, "bpi20_domestic", "train_test", "test.csv")
#     )
#     train["split"] = "train"
#     test["split"] = "test"
#     log = pd.concat([train, test])
#     return log


# @cache
# @common_preprocessing
# def bpi20_international(path=None):
#     path = PATH_ROOT if path is None else path
#     log_path = os.path.join(path, "bpi20_international")
#     log = read_log(log_path)

#     train_set, test_set = pm4py.split_train_test(log.copy())
#     train_set.loc[:, "split"] = "train"
#     test_set.loc[:, "split"] = "test"
#     log = pd.concat([train_set, test_set])
#     return log


@cache
@common_preprocessing
def bpi20_permit():
    train = pd.read_csv(
        os.path.join(PATH_ROOT, "bpi20_permit", "train_test", "train.csv")
    )
    test = pd.read_csv(
        os.path.join(PATH_ROOT, "bpi20_permit", "train_test", "test.csv")
    )
    train["split"] = "train"
    test["split"] = "test"
    log = pd.concat([train, test])
    return log


@cache
@common_preprocessing
def bpi20_prepaid():
    train = pd.read_csv(
        os.path.join(PATH_ROOT, "bpi20_prepaid", "train_test", "train.csv")
    )
    test = pd.read_csv(
        os.path.join(PATH_ROOT, "bpi20_prepaid", "train_test", "test.csv")
    )
    train["split"] = "train"
    test["split"] = "test"
    log = pd.concat([train, test])
    return log


@cache
@common_preprocessing
def sepsis(path=None):
    path = PATH_ROOT if path is None else path
    log_path = os.path.join(path, "sepsis")
    log = read_log(log_path)

    train_set, test_set = pm4py.split_train_test(log.copy())
    train_set.loc[:, "split"] = "train"
    test_set.loc[:, "split"] = "test"
    log = pd.concat([train_set, test_set])
    log = log.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp",
    })

    activities = log.activity.value_counts()
    activities = activities[activities >= 0.05 * len(log)].index.values
    log = log[log.activity.isin(activities)]
    return log


@cache
@common_preprocessing
def bpi15():
    train = pd.read_csv(os.path.join(PATH_ROOT, "bpi15", "train_test", "train.csv"))
    test = pd.read_csv(os.path.join(PATH_ROOT, "bpi15", "train_test", "test.csv"))
    train["split"] = "train"
    test["split"] = "test"
    log = pd.concat([train, test])
    return log
