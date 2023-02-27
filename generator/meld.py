""" 
Managing Event Log Datasets for deep learning
"""

import pandas as pd
import numpy as np


def index_of_ngrams(log, prefix_len=5):
    from nltk import ngrams

    log = log.reset_index(drop=True)

    def f_ngram(group_ids):
        # indicies of ngrams for each group (case)
        return list(
            ngrams(group_ids, n=prefix_len, pad_left=True, left_pad_symbol=0)
        )  # 0 is the reference index for padding

    cases_ngrams = log.reset_index().groupby("case_id")["index"].apply(f_ngram).values
    import functools
    import operator

    # list of list of elements to list of elements (elements are tuples in this case)
    cases_ngrams = functools.reduce(operator.iconcat, cases_ngrams, [])
    # converting tuples to list
    cases_ngrams = list(map(list, cases_ngrams))
    return cases_ngrams


def prepare_log(log):
    """Prepares an event log before transforming it into a ngram model.

    This function takes as input a pd.DataFrame and applies basic offset operations
    needed to index cases in a n-gram fashion.

    ToDo: apply the word_to_ix outside here.

    Args:
        log (pd.DataFrame): Event log

    Returns:
        pd.DataFrame: Offsetted event log for managing cases and use ngrams.
    """
    exclude_cols = ["case_id", "time", "remaining_time", "type_set", "target"]
    categorical = log.select_dtypes(include=["object"]).columns.difference(exclude_cols)

    # adding an ending offset to cases
    end_of_seq = log.drop_duplicates("case_id", keep="last").reset_index(drop=True)
    for c in categorical:
        end_of_seq.loc[:, c] = "<eos>"

    end_of_seq.time += pd.tseries.offsets.Minute()
    log = pd.concat((log, end_of_seq))

    # sorting by case_id and time
    log.sort_values(["case_id", "time"], inplace=True)

    # we need to create a reference index for padding
    # as our sequential natural over cases is different from time series and nlp, we are creating ngrams of indexes to
    # slide over cases instead of the whole log;
    # e.g. starting traces might be (0, 0, 1) or (0, 0, 32); whre 1 and 32 are indexes for first events of two different cases,
    # whereas the 0 index is a row of 0s or nans
    zero_event_index = pd.DataFrame(
        data={
            "case_id": None,
            "activity": "<pad>",
            "time": pd.NaT,
            "resource": "<pad>" if "resource" in categorical else 0,
            "remaining_time": 0,
            "type_set": "<pad>",
            "target": 0,
        },
        index=[0],
    )
    zero_event_index["time"] = pd.to_datetime(zero_event_index["time"], utc=True)
    log = pd.concat((zero_event_index, log))
    log.reset_index(drop=True, inplace=True)
    return log


def vectorize_log(
    log, include_cols=["activity", "resource", "remaining_time"], prefix_len=5
):
    train_ngrams_ix = index_of_ngrams(
        log[log["type_set"] == "train"], prefix_len=prefix_len
    )
    test_ngrams_ix = index_of_ngrams(
        log[log["type_set"] == "test"], prefix_len=prefix_len
    )

    train_events = log.loc[
        log["type_set"].isin(["<pad>", "train"]), include_cols
    ].values
    test_events = log.loc[log["type_set"].isin(["<pad>", "test"]), include_cols].values

    train_condition = log.loc[log["type_set"].isin(["<pad>", "train"]), "target"].values
    test_condition = log.loc[log["type_set"].isin(["<pad>", "train"]), "target"].values

    train = (
        train_events,
        train_condition,
        train_ngrams_ix[1:],
    )  # first index (0-index) regards the pad row, so we just ignore it
    test = (test_events, test_condition, test_ngrams_ix)
    # ToDo: refactor this function
    # train = train[1:]
    # test = test[1:]     # ignoring index-0: pad reference
    return train, test
