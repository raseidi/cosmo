import multiprocessing
from typing import List, Union, Tuple

import torch
from pandas import DataFrame
from torch import nn
from torch.utils.data import Dataset
from collections import OrderedDict

from cosmo.utils import ensure_dir


class EventLogDataset(Dataset):
    _cases: list[str] = None
    _num_features: int = None
    _num_cat_features: int = None
    _num_cont_features: int = None
    PAD_IDX: int = 0
    UNK_IDX: int = 1
    # EOS_IDX: int = 2

    def __init__(
        self,
        log: DataFrame,
        vocab: Tuple[dict, dict] = None,
        categorical_features: List[str] = ["activity"],
        continuous_features: List[str] = None,
        target: str = "activity",
        dataset_name: str = None,
        train: bool = True,
    ):
        self.log = log.copy().reset_index(drop=True)
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.target = target
        self.train = train
        self.dataset_name = dataset_name
        self.set_vocab(vocab)

    def set_vocab(self, vocab):
        if vocab is None:
            self.feature2idx = OrderedDict()
            self.idx2feature = OrderedDict()
            for feature in self.categorical_features:
                unique_values = sorted(self.log[feature].unique())
                self.feature2idx[feature] = {
                    value: i + 2 for i, value in enumerate(unique_values)
                }
                self.idx2feature[feature] = {
                    i + 2: value for i, value in enumerate(unique_values)
                }
                self.feature2idx[feature].update(
                    {
                        "<PAD>": self.PAD_IDX,
                        # "<EOS>": self.EOS_IDX, # we already have EOS in the log preprocessing
                        "<UNK>": self.UNK_IDX,
                    }
                )
                self.idx2feature[feature].update(
                    {
                        self.PAD_IDX: "<PAD>",
                        # self.EOS_IDX: "<EOS>",
                        self.UNK_IDX: "<UNK>",
                    }
                )
        else:
            self.feature2idx, self.idx2feature = vocab

    def __getitem__(self, index):
        raise "Not implemented"

    def __len__(self):
        raise "Not implemented"


class ContinuousTraces(Dataset):
    """log for continuous event traces

    This log is used for continuous traces.
    The log is a list of complete traces.

    Args:
        log (DataFrame): a dataframe of traces
        categorical_features (List[str]): a list of categorical features
        continuous_features (List[str]): a list of continuous features
    """

    _cases: list[str] = None
    _num_features: int = None
    _num_cat_features: int = None
    _num_cont_features: int = None
    PAD_IDX: int = 0
    UNK_IDX: int = 1
    # EOS_IDX: int = 2

    def __init__(
        self,
        log: DataFrame,
        vocab: Tuple[dict, dict] = None,
        categorical_features: List[str] = ["activity"],
        continuous_features: List[str] = None,
        target: str = "activity",
        dataset_name: str = None,
        train: bool = True,
    ):
        self.log = log
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features if continuous_features else []
        self.target = target
        self.train = train
        self.dataset_name = dataset_name
        self.set_vocab(vocab)
        self.dataset = self._build_dataset()

    def set_vocab(self, vocab):
        if vocab is None:
            self.feature2idx = OrderedDict()
            self.idx2feature = OrderedDict()
            for feature in self.categorical_features:
                unique_values = sorted(self.log[feature].unique())
                self.feature2idx[feature] = {
                    value: i + 2 for i, value in enumerate(unique_values)
                }
                self.idx2feature[feature] = {
                    i + 2: value for i, value in enumerate(unique_values)
                }
                self.feature2idx[feature].update(
                    {
                        "<PAD>": self.PAD_IDX,
                        # "<EOS>": self.EOS_IDX, # we already have EOS in the log preprocessing
                        "<UNK>": self.UNK_IDX,
                    }
                )
                self.idx2feature[feature].update(
                    {
                        self.PAD_IDX: "<PAD>",
                        # self.EOS_IDX: "<EOS>",
                        self.UNK_IDX: "<UNK>",
                    }
                )
        else:
            self.feature2idx, self.idx2feature = vocab

    def get_vocabs(self):
        """This method is necessary for the test set to
        use the same vocab as the training set"""
        return (self.feature2idx, self.idx2feature)

    @property
    def cases(self):
        if self._cases is None:
            self._cases = self.log["case_id"].unique()
        return self._cases

    @property
    def num_cont_features(self):
        if self._num_cont_features is None:
            self._num_cont_features = len(self.continuous_features)
        return self._num_cont_features

    @property
    def num_cat_features(self):
        if self._num_cont_features is None:
            self._num_cont_features = len(self.categorical_features)
        return self._num_cont_features

    @property
    def num_features(self):
        if self._num_features is None:
            self._num_features = len(self.continuous_features) + len(
                self.categorical_features
            )
        return self._num_features

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_id = self.cases[idx]
        return self.dataset[case_id]

    def get_case(self, case_id: str):
        case_id = list(self.cases).index(case_id)
        return self[case_id]

    # @cache(name="dataset", format="json")
    # this is super slow, gotta fix it later; for now, we'll just cache the tensors in the disk
    def _build_dataset(self):
        import os

        # ToDo: cache properly
        split = "train" if self.train else "test"

        if len(self.dataset_name.split("_")) == 2:
            dataset_name, template = self.dataset_name.split("_")
        else:
            a, b, template = self.dataset_name.split("_")
            dataset_name = a + "_" + b
        path = f"data/{dataset_name}/cached_train_test/dataset_{template}_{split}.pt"
        if os.path.exists(path):
            tensors = torch.load(path)
        else:
            ensure_dir(os.path.dirname(path))
            tensors = {}
            data = self.log.copy()
            for cat in self.categorical_features:
                data[cat] = data[cat].map(self.feature2idx[cat], na_action="ignore")
                data[cat] = data[cat].fillna(self.UNK_IDX)

            for case_id in self.cases:
                tensors[case_id] = {}
                trace = data[data["case_id"] == case_id]
                for feature in self.categorical_features:
                    print(len(tensors))
                    values = trace[feature].values
                    tensors[case_id][f"cat_{feature}"] = torch.tensor(
                        values[:-1],
                        dtype=torch.long,
                    )
                    if feature == self.target:
                        tensors[case_id]["target"] = torch.tensor(
                            values[1:],
                            dtype=torch.long,
                        )
                for feature in self.continuous_features:
                    values = trace[feature].values[:-1]
                    tensors[case_id][f"num_{feature}"] = torch.tensor(
                        values, dtype=torch.float
                    )

            torch.save(tensors, path)
            del data
        return tensors

    def get_stoi(self):
        return self.feature2idx

    def get_itos(self):
        return self.idx2feature


class ConstrainedContinuousTraces(ContinuousTraces):
    _num_constraints: int = None

    def __init__(
        self,
        log: DataFrame,
        constraints: DataFrame,
        vocab: Tuple[dict, dict] = None,
        categorical_features: List[str] = ["activity"],
        continuous_features: List[str] = None,
        dataset_name: str = None,
        train: bool = True,
        device: str = None,
    ):
        super().__init__(
            log=log,
            vocab=vocab,
            categorical_features=categorical_features,
            continuous_features=continuous_features,
            dataset_name=dataset_name,
            train=train,
        )
        self.constraints = constraints
        self.dataset_name = dataset_name
        self.train = train
        self.device = device
        self._validate()
        self.dataset = self._build_constrained_dataset()

    def _validate(self):
        if set(self.log.case_id) - set(self.constraints.case_id):
            raise ValueError("Some cases are missing in the constraints")

        self.constraints.set_index("case_id", inplace=True)

    def __getitem__(self, idx: int):
        # making it explict but we don't need to override this method
        return super().__getitem__(idx)

    def _build_constrained_dataset(self):
        import os

        # ToDo: cache properly
        split = "train" if self.train else "test"

        if len(self.dataset_name.split("_")) == 2:
            dataset_name, template = self.dataset_name.split("_")
        else:
            a, b, template = self.dataset_name.split("_")
            dataset_name = a + "_" + b

        path = f"data/{dataset_name}/cached_train_test/dataset_{template}_{split}.pt"
        if os.path.exists(path):
            self.dataset = torch.load(path)
        ensure_dir(os.path.dirname(path))
        if "constraints" not in self.dataset[self.log.case_id.unique()[0]]:
            for case_id in self.cases:
                trace = self.constraints[self.constraints.index == case_id]
                self.dataset[case_id]["constraints"] = torch.tensor(
                    trace.values, dtype=torch.float32
                )

            torch.save(self.dataset, path)

        if self.device is not None:
            for case_id in self.cases:
                for feature in self.dataset[case_id]:
                    self.dataset[case_id][feature] = self.dataset[case_id][feature].to(
                        self.device
                    )
        return self.dataset

    @property
    def num_constraints(self):
        if self._num_constraints is None:
            self._num_constraints = self.constraints.shape[1]
        return self._num_constraints


# class PrefixTraces(EventLogDataset):
#     def __init__(
#         self,
#         log: DataFrame,
#         constraints: DataFrame,
#         vocab: Tuple[dict, dict] = None,
#         categorical_features: List[str] = ["activity"],
#         continuous_features: List[str] = None,
#         dataset_name:str = None,
#         train: bool = True,
#         device: str = None,
#         prefix_len: int = 5,
#     ):
#         super().__init__(
#             log=log,
#             vocab=vocab,
#             categorical_features=categorical_features,
#             continuous_features=continuous_features,
#             dataset_name=dataset_name,
#             train=train
#         )
#         self.prefix_len = prefix_len
#         self._build_dataset()

#     def __getitem__(self, index: int):
#         pass

#     def __len__(self) -> int:
#         pass

#     def _build_dataset(self):
#         import pandas as pd
#         pad_row = self.log.iloc[0].copy()
#         pad_row["case_id"] = "PAD"
#         for feature in self.categorical_features + self.continuous_features:
#             pad_row[feature] = self.PAD_IDX
#         self.log = pd.concat((pad_row, self.log))
#         ngrams_ix = self.index_of_ngrams(
#             self.log, prefix_len=self.prefix_len
#         )
#         events = self.log.loc[
#             :,
#             self.categorical_features + self.continuous_features,
#         ].values
#         # return events, condition, ngrams_ix

#         train = (
#             events,
#             # train_condition,
#             ngrams_ix[1:],
#         )  # first index (0-index) regards the pad row, so we just ignore it
#         # ToDo: refactor this function
#         # train = train[1:]
#         # test = test[1:]     # ignoring index-0: pad reference
#         return train

#     def index_of_ngrams(key, log):
#         prefix_len=5 # TODO FIX with starmap
#         from nltk import ngrams
#         def f_ngram(key, group_ids):
#             if len(group_ids) < prefix_len:
#                 # fill with zeros
#                 group_ids = list(group_ids.values)
#                 group_ids += [0] * (prefix_len - len(group_ids))
#             return list(
#                 ngrams(group_ids, n=prefix_len)
#             )  # 0 is the reference index for padding

#         group = log[log.case_id=="AKA"].reset_index().groupby("case_id")["index"]
#         with multiprocessing.Pool(20) as pool:
#             cases_ngrams = pool.starmap(f_ngram, group)

#         cases_ngrams
#         group.size().idxmin()
#         group.last()
#         log.loc[cases_ngrams[0][0],:]

#         import functools
#         import operator

#         # list of list of elements to list of elements (elements are tuples in this case)
#         cases_ngrams = functools.reduce(operator.iconcat, cases_ngrams, [])
#         # converting tuples to list
#         cases_ngrams = list(map(list, cases_ngrams))
#         return cases_ngrams
