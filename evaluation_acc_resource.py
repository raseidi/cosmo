""" 
I forgot to log the accuracy of resources at wandb 
so I'm doing this script to solve this issue. It won't
be used in the future.
"""
import os
import torch
import numpy as np
import pandas as pd
from itertools import product
from argparse import Namespace
from cosmo import MTCondLSTM, MTCondDG
from cosmo.data_loader import get_loader
from cosmo.meld import prepare_log, vectorize_log
from cosmo.utils import get_runs, load_checkpoint, get_vocabs, read_data


def next_evt_pred(model, test_loader, device="cuda"):
    ac_pred = []
    ac_true = []
    rt_true, rt_pred = [], []
    res_true, res_pred = [], []
    with torch.inference_mode():
        for e, (X, y) in enumerate(test_loader):
            x = [d.to(device) for d in X]
            y = [d.to(device) for d in y]

            na, res, rt, _ = model(x)

            ac_true.extend(y[0].cpu().detach().tolist())
            res_true.extend(y[1].cpu().detach().tolist())
            rt_true.extend(y[2].cpu().detach().tolist())

            ac_pred.extend(
                torch.argmax(torch.softmax(na, dim=1), dim=1).cpu().detach().tolist()
            )
            res_pred.extend(
                torch.argmax(torch.softmax(res, dim=1), dim=1).cpu().detach().tolist()
            )
            rt_pred.extend(rt.reshape(-1).cpu().detach().tolist())

    return {
        "ac_true": ac_true,
        "ac_pred": ac_pred,
        "res_true": res_true,
        "res_pred": res_pred,
        "rt_true": rt_true,
        "rt_pred": rt_pred,
    }


def best_to_dict(df, dataset, condition, model="DG"):
    res = df[(df.dataset == dataset) & (df.condition == condition) & (df.model == model)].iloc[0, :].to_dict()
    res = {
        key: value
        for key, value in res.items()
        if not key.startswith("train")
        and not key.startswith("test")
        and not key.startswith("_")
    }
    return res


if __name__ == "__main__":
    datasets = pd.read_csv(os.path.join("results", "datasets_statistics.csv"))[
        "dataset"
    ].unique()
    conditions = ["resource_usage"]
    models = ["DG"]
    ignore_datasets = [
        "BPI_Challenge_2012_W",
        "BPI_Challenge_2012_Complete",
        "BPI_Challenge_2012_W_Complete",
        "BPI_Challenge_2012_A",
        "BPI_Challenge_2012_O",
        "BPI_Challenge_2012",
    ]
    final = pd.read_csv("results/predictions.csv")
    prods = product(datasets, conditions, models)
    for dataset, condition, model_arc in prods:
        if dataset in ignore_datasets and condition == "resource_usage":
            continue
        bpm_results = pd.read_csv(os.path.join("results", "best_runs.csv"))
        params = best_to_dict(bpm_results, dataset, condition, model_arc)
        if params is None:
            print(dataset, condition, model_arc, "Not found")
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
        if model_arc == "":
            model = MTCondLSTM(vocabs=vocabs, batch_size=params.batch_size)
        elif model_arc == "DG":
            model = MTCondDG(vocabs=vocabs, batch_size=params.batch_size)
        
        checkpoint = load_checkpoint(
            ckpt_dir_or_file=f"models/{params.dataset}/{params.condition}/{params.run_name}/best_model.ckpt"
        )
        model.load_state_dict(checkpoint["net"])
        model.cuda()
        model.eval()
        preds = next_evt_pred(model, test_loader, device="cuda")
        preds = pd.DataFrame(preds)
        preds["dataset"] = dataset
        preds["condition"] = condition
        preds["model"] = model_arc
        final = pd.concat((final, preds))
        final.to_csv("results/predictions.csv", index=False)
