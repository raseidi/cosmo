import os
import pandas as pd
from itertools import product
from argparse import Namespace

from cosmo import MTCondLSTM, MTCondDG
from cosmo.data_loader import get_loader
from cosmo.meld import prepare_log, vectorize_log
from cosmo.simulator.simulator import simulate_remaining_case
from cosmo.utils import ensure_dir, get_runs, load_checkpoint, get_vocabs, read_data


def read_tuning_results() -> pd.DataFrame:
    """Returns a dataframe containing all the runs from wandb.

    If the runs are not available locally it reads it from wandb.
    ToDo: make wandb project public so anyone can read the runs.

    Returns:
        runs: pd.DataFrame
    """
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
    """
    Function to ensure a model has been trained for a given dataset.

    If the `run` at wandb exists, continue. Otherwise, train the model from scratch.
    """
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

if __name__ == "__main__":
    ensure_dir("results/")
    ensure_dir("results/simulations")
    ensure_dir("results/datasets")
    datasets = [
        "PrepaidTravelCost",
        "PermitLog",
        "bpi17",
        "bpi_challenge_2013_incidents",
        "BPI_Challenge_2013_closed_problems",
        "RequestForPayment",
        "bpi19",
    ]
    conditions = ["resource_usage"]

    model_arcs = ["Baseline", "DG"]  # "" refers to the baseline
    prods = product(datasets, conditions, model_arcs)
    for dataset, condition, model_arc in prods:
        ensure_model(dataset=dataset, condition=condition, model=model_arc)

        bpm_results = pd.read_csv(os.path.join("results", "best_runs.csv"))
        params = best_to_dict(bpm_results, dataset, condition)
        params = Namespace(**params)

        log = read_data(os.path.join("data", params.dataset, "log.csv"))
        log["target"] = log[params.condition]
        log.drop(["trace_time", "resource_usage", "variant"], axis=1, inplace=True)
        log = prepare_log(log)
        vocabs = get_vocabs(log=log)

        for f in vocabs:
            log.loc[:, f] = log.loc[:, f].transform(lambda x: vocabs[f]["stoi"][x])

        # no need to load train loader here
        _, data_test = vectorize_log(log)
        test_loader = get_loader(data_test, batch_size=1024, shuffle=False)
        if model_arc == "Baseline":
            model = MTCondLSTM(vocabs=vocabs, batch_size=params.batch_size)
        elif model_arc == "DG":
            model = MTCondDG(vocabs=vocabs, batch_size=params.batch_size)
        else:
            continue
        checkpoint = load_checkpoint(
            ckpt_dir_or_file=f"models/{params.dataset}/{params.condition}/{params.run_name}/best_model.ckpt"
        )
        model.load_state_dict(checkpoint["net"])
        model.cuda()
        model.eval()

        itos = {value: key for key, value in vocabs["activity"]["stoi"].items()}
        if "resource" in vocabs:
            ritos = {value: key for key, value in vocabs["resource"]["stoi"].items()}

        if not os.path.exists(
            f"results/simulations/{params.dataset}_{params.condition}_{model_arc}.csv"
        ):
            on_going = simulate_remaining_case(model, log[log.type_set == "test"])
            on_going.activity = on_going.activity.apply(lambda x: itos[x])
            if "resource" in vocabs:  # i.e. resource is numerical
                on_going.resource = on_going.resource.apply(lambda x: ritos[x])

            on_going.to_csv(
                f"results/simulations/{params.dataset}_{params.condition}_{model_arc}.csv",
                index=False,
            )
        break
