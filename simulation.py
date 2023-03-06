import os
import torch
import pandas as pd
from itertools import product
from argparse import Namespace
from generator import MTCondLSTM
from generator.data_loader import get_loader
from generator.meld import prepare_log, vectorize_log
from generator.simulator.simulator import simulate_from_scratch, simulate_remaining_case
from generator.utils import get_runs, load_checkpoint, get_vocabs, read_data


def read_tuning_results():
    if os.path.exists("results/results.csv"):
        runs = pd.read_csv("results/results.csv")
    else:  # wandb api sometimes takes a while to load everything
        runs = get_runs("multi-task")
        runs.to_csv("results/results.csv", index=False)
    return runs


def best_to_dict(df, dataset, condition):
    res = df[(df.dataset == dataset) & (df.condition == condition)].iloc[0, :].to_dict()
    res = {
        key: value
        for key, value in res.items()
        if not key.startswith("train")
        and not key.startswith("test")
        and not key.startswith("_")
    }
    return res


def get_best_run(dataset: str, condition: str, metric="test_loss"):
    # w.r.t to loss of the NA
    top_runs_ix = runs.groupby(["dataset", "condition"])[metric].idxmin().values
    df = runs.iloc[top_runs_ix, :]
    res = best_to_dict(df, dataset, condition)
    res.pop('run_name', None)
    return res


"""
1. read bpm23 project
2. check if dataset model exists, otherwise read results and get best run
    2.1. retrain the model from the best run
3. load the model checkpoint
4. simulate from scratch
5. simulate on going cases
"""
datasets = pd.read_csv("results/datasets_statistics.csv")["dataset"].unique()
conditions = ["resource_usage", "trace_time"]
prods = product(datasets, conditions)

# condition = "resource_usage"
# dataset = "RequestForPayment"
for dataset, condition in prods:
    try:
        bpm_results = pd.read_csv("results/bpm_performances.csv")
    except:
        bpm_results = None
    if (
        bpm_results is None
        or bpm_results[
            (bpm_results["dataset"] == dataset) & (bpm_results["condition"] == condition)
        ].empty
    ):
        runs = read_tuning_results()
        res = get_best_run(dataset=dataset, condition=condition)

        from train import main

        params = Namespace(**res)
        params.project_name = "bpm23"
        main(params)
        bpm_results = get_runs("bpm23")
        bpm_results.to_csv("results/bpm_performances.csv", index=False)

# params = best_to_dict(bpm_results, dataset, condition)
# params = Namespace(**params)

# log = read_data(f"data/{params.dataset}/log.csv")
# log["target"] = log[params.condition]
# log.drop(["trace_time", "resource_usage", "variant"], axis=1, inplace=True)
# log = prepare_log(log)
# vocabs = get_vocabs(log=log)

# for f in vocabs:
#     log.loc[:, f] = log.loc[:, f].transform(lambda x: vocabs[f]["stoi"][x])

# # no need to load train loader here
# _, data_test = vectorize_log(log)
# test_loader = get_loader(data_test, batch_size=1024, shuffle=False)

# model = MTCondLSTM(vocabs=vocabs, batch_size=params.batch_size)
# checkpoint = load_checkpoint(
#     ckpt_dir_or_file=f"models/{params.dataset}/{params.condition}/{params.run_name}/best_model.ckpt"
# )
# model.load_state_dict(checkpoint["net"])
# model.cuda()
# model.eval()

# stats = pd.read_csv("results/datasets_statistics.csv")
# max_len_trace = stats[stats.dataset == dataset]["len_trace_max"].values[0]
# sim = simulate_from_scratch(model, vocabs, n_traces=5, max_len=max_len_trace)
# sim[sim["case_id"] == 1]
# # sim = simulate_remaining_case(model, log)
# # sim.to_csv(f"evaluation/simulation/{dataset}/from_scratch.csv", index=False)
