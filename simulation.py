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
from generator.preprocessing import label_variants


def get_variants(df):
    variants = df.groupby(["case_id"])["activity"].apply(
        list
    )  # transform groupby into list
    variants = variants.apply(
        lambda x: ",".join(map(str, x))
    )  # transfor list into a unique string
    return sorted(set(variants))


def read_tuning_results():
    if os.path.exists("results/all_runs.csv"):
        runs = pd.read_csv("results/all_runs.csv")
    else:  # wandb api sometimes takes a while to load everything
        runs = get_runs("multi-task")
        runs.to_csv("results/all_runs.csv", index=False)
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


def get_best_run(dataset: str, condition: str, runs, metric="test_loss"):
    # w.r.t to loss of the NA
    top_runs_ix = runs.groupby(["dataset", "condition"])[metric].idxmin().values
    df = runs.iloc[top_runs_ix, :]
    res = best_to_dict(df, dataset, condition)
    res.pop("run_name", None)
    return res


def ensure_model(dataset, condition, model):
    try:
        bpm_results = pd.read_csv("results/best_runs.csv")
    except:
        bpm_results = None
    if (
        bpm_results is None
        or bpm_results[
            (bpm_results["dataset"] == dataset)
            & (bpm_results["condition"] == condition)
            & (bpm_results["model"] == model)
        ].empty
    ):
        runs = read_tuning_results()
        res = get_best_run(dataset=dataset, condition=condition, runs=runs)

        from train import main

        params = Namespace(**res)
        params.project_name = "bpm23"
        params.model = model
        main(params)
        print(dataset, condition)
        bpm_results = get_runs("bpm23")
        bpm_results.to_csv("results/best_runs.csv", index=False)


"""
1. read bpm23 project
2. check if dataset model exists, otherwise read results and get best run
    2.1. retrain the model from the best run
3. load the model checkpoint
4. simulate from scratch
5. simulate on going cases
"""
datasets = pd.read_csv(os.path.join("results", "datasets_statistics.csv"))[
    "dataset"
].unique()
conditions = ["resource_usage"]
prods = product(datasets, conditions)
data_stats = pd.read_csv("results/datasets_statistics.csv")
ignore_datasets = [
    "BPI_Challenge_2012_W",
    "BPI_Challenge_2012_Complete",
    "BPI_Challenge_2012_W_Complete",
    "BPI_Challenge_2012_A",
    "BPI_Challenge_2012_O",
    "BPI_Challenge_2012",
]

# dataset = "bpi19"
# condition = "resource_usage"
# model = "baseline"
model = "DG"
for dataset, condition in prods:
    if dataset in ignore_datasets and condition == "resource_usage":
        continue
    dataset = "RequestForPayment"
    ensure_model(dataset=dataset, condition=condition, model=model)

    bpm_results = pd.read_csv(os.path.join("results", "best_runs.csv"))
    params = best_to_dict(bpm_results, dataset, condition)
    params = Namespace(**params)
    # print(params)
    # params = Namespace(
    #     batch_size=64,
    #     condition="trace_time",
    #     dataset="bpi_challenge_2013_incidents",
    #     device="cuda",
    #     epochs=50,
    #     lr=0.0007648621728067,
    #     optimizer="adam",
    #     project_name="bpm23",
    #     run_name="tough-smoke-26",
    #     weight_decay=0.0,
    # )

    log = read_data(os.path.join("data", params.dataset, "log.csv"))
    log["target"] = log[params.condition]
    log.drop(["trace_time", "resource_usage", "variant"], axis=1, inplace=True)
    log = prepare_log(log)
    vocabs = get_vocabs(log=log)
    if "resource" not in vocabs:  # i.e. resource is numerical
        # z score normalization for numerical resource
        mean = log.loc[log.type_set == "train", "resource"].mean()
        std = log.loc[log.type_set == "train", "resource"].std()
        log.loc[:, "resource"] = (log.loc[:, "resource"] - mean) / std

    for f in vocabs:
        log.loc[:, f] = log.loc[:, f].transform(lambda x: vocabs[f]["stoi"][x])

    # no need to load train loader here
    _, data_test = vectorize_log(log)
    test_loader = get_loader(data_test, batch_size=1024, shuffle=False)
    model = MTCondLSTM(vocabs=vocabs, batch_size=params.batch_size)
    checkpoint = load_checkpoint(
        ckpt_dir_or_file=f"models/{params.dataset}/{params.condition}/{params.run_name}/best_model.ckpt"
    )
    model.load_state_dict(checkpoint["net"])
    model.cuda()
    model.eval()

    # max_len_trace = data_stats[data_stats.dataset == params.dataset][
    #     "len_trace_max"
    # ].values[0]
    # it's gonna generate half for each condition
    # n_traces = 5 # int(log[log.type_set == "test"].case_id.nunique() / 2)

    itos = {value: key for key, value in vocabs["activity"]["stoi"].items()}
    if "resource" in vocabs:
        ritos = {value: key for key, value in vocabs["resource"]["stoi"].items()}
    # if not os.path.exists(
    #     f"results/simulations/{params.dataset}_{params.condition}_from_scratch.csv"
    # ):
    #     from_scratch = simulate_from_scratch(
    #         model, n_traces=n_traces, max_len=max_len_trace
    #     )
    #     from_scratch.activity = from_scratch.activity.apply(lambda x: itos[x])
    #     if "resource" not in vocabs:  # i.e. resource is numerical
    #         # z score normalization for numerical resource
    #         from_scratch.loc[:, "resource"] = (
    #             from_scratch.loc[:, "resource"] * std + mean
    #         )

    #     from_scratch.to_csv(
    #         f"results/simulations/{params.dataset}_{params.condition}_from_scratch.csv",
    #         index=False,
    #     )

    if not os.path.exists(
        f"results/simulations/{params.dataset}_{params.condition}_on_going.csv"
    ):
        on_going = simulate_remaining_case(model, log[log.type_set == "test"])
        on_going.activity = on_going.activity.apply(lambda x: itos[x])
        if "resource" not in vocabs:  # i.e. resource is numerical
            # z score normalization for numerical resource
            on_going.loc[:, "resource"] = on_going.loc[:, "resource"] * std + mean
        else:
            on_going.resource = on_going.resource.apply(lambda x: ritos[x])

        on_going.to_csv(
            f"results/simulations/{params.dataset}_{params.condition}_{model}_on_going.csv",
            index=False,
        )
    break
