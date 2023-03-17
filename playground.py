import torch
from torch import nn

from generator.meld import vectorize_log, prepare_log
from generator.data_loader import get_loader
from generator import MTCondLSTM, MTCondDG
from generator.training import train
from generator.utils import get_runs, get_vocabs, read_data


file = "data/RequestForPayment/log.csv"
log = read_data(file)
log["target"] = log["resource_usage"]
log.drop(["trace_time", "resource_usage", "variant"], axis=1, inplace=True)
log = prepare_log(log)
vocabs = get_vocabs(log)
for f in vocabs:
    log.loc[:, f] = log.loc[:, f].transform(lambda x: vocabs[f]["stoi"][x])
data_train, data_test = vectorize_log(log)
test_loader = get_loader(data_test, batch_size=3, shuffle=False)

model = MTCondDG(vocabs, 3)
x, y = next(iter(test_loader))
a1, a2, a3, states = model(x)
a1

import pandas as pd

df = pd.read_csv("results/best_runs.csv")
df["model"] = "baseline"
df.to_csv("results/best_runs.csv", index=False)
