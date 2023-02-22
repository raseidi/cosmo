from argparse import Namespace
import torch
import wandb
import pandas as pd

from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from generator.meld import vectorize_log, prepare_log
from generator.data_loader import get_loader
from generator.models import MTCondLSTM
from generator.training import train
from generator.utils import get_runs, read_data


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Pytorch Implementation for Condition-based Trace Generator",
        add_help=add_help,
    )

    parser.add_argument(
        "--dataset",
        default="RequestForPayment",
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--condition",
        default="trace_time",
        type=str,
        help="condition",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        help="batch size, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--lr", default=5e-3, type=float, help="initial learning rate")

    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: None)",
        dest="weight_decay",
    )

    parser.add_argument(
        "--scheduler",
        default=None,
        type=str,
        help="the lr scheduler (default: steplr)",
    )

    return parser


def get_vocabs(log, features=["activity", "resource"]):
    vocabs = dict()
    for f in features:
        stoi = {v: k for k, v in enumerate(log.loc[:, f].unique())}
        vocabs[f] = {
            "stoi": stoi,
            "size": len(stoi),
            "emb_dim": int(len(stoi) ** (1 / 2))
            if int(len(stoi) ** (1 / 2)) > 2
            else 2,
        }
    return vocabs


def experiment_exists(params, run_name):
    try:
        runs_df = get_runs(run_name)
    except:
        return False
    try:
        return runs_df[
            (runs_df["dataset"] == params.dataset)
            & (runs_df["condition"] == params.condition)
            & (runs_df["learning_rate"] == params.lr)
            & (runs_df["epochs"] == params.epochs)
            & (runs_df["weight_decay"] == params.weight_decay)
            & (runs_df["batch_size"] == params.batch_size)
            & (runs_df["device"] == params.device)
        ].empty
    except:
        return False



def main(params):
    print(params)
    run_name = f"multi-task-{params.condition.replace('_', '-')}"
    # if experiment_exists(params, run_name):
    #     raise Exception("Experiment has already been done.")

    log = read_data(f"data/{params.dataset}/{params.condition}/log.csv")
    log = prepare_log(log)
    vocabs = get_vocabs(log)
    # encoding
    for f in vocabs:
        log.loc[:, f] = log.loc[:, f].transform(lambda x: vocabs[f]["stoi"][x])

    # ToDo how to track cat features? here we have (act, res, rt)
    data_train, data_test = vectorize_log(log)

    train_loader = get_loader(data_train, batch_size=params.batch_size)
    test_loader = get_loader(data_test, batch_size=1000, shuffle=False)

    torch.manual_seed(0)
    model = MTCondLSTM(vocabs=vocabs, batch_size=params.batch_size)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0.5)

    model.apply(init_weights)

    model.to(params.device)
    # X, y = next(iter(train_loader))
    # model(X)

    # logger = wandb.init(project="multi-task-time-condition")
    logger = wandb.init(project=run_name)
    logger.config.update(
        {
            "dataset": params.dataset,
            "condition": params.condition,
            "batch_size": params.batch_size,
            "epochs": params.epochs,
            "learning_rate": params.lr,
            "weight_decay": params.weight_decay,
            "scheduler": params.scheduler,
            "device": params.device,
        }
    )
    wandb.watch(model, log="all")

    criterion = {"clf": nn.CrossEntropyLoss(), "reg": nn.MSELoss()}
    optm = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    if params.scheduler:
        sc = MultiStepLR(optm, milestones=[25, 35], gamma=0.1)
    else:
        sc = None
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=criterion,
        optimizer=optm,
        sc=sc,
        logger=logger,
    )
    logger.finish()


if __name__ == "__main__":
    params = get_args_parser().parse_args()
    # print(params)
    main(params)
