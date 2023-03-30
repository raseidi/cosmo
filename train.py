"""
Script for tuning models using WandB Sweeps.
"""

import torch
import wandb
import pprint

from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from cosmo.meld import vectorize_log, prepare_log
from cosmo.data_loader import get_loader
from cosmo import MTCondLSTM, MTCondDG
from cosmo.training import train
from cosmo.utils import get_runs, get_vocabs, read_data


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Pytorch Implementation for CoSMo",
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
        default="resource_usage",
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
        "--lr",
        default="0.0001",
        type=float,
        help="Learning rate",
    )

    parser.add_argument(
        "--epochs",
        default="50",
        type=int,
        help="Number of epochs",
    )

    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        help="Optimizer (adam or sgd)",
    )

    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Batch size",
    )

    parser.add_argument(
        "--weight-decay",
        default=0,
        type=float,
        help="Weight decay",
    )

    parser.add_argument(
        "--project-name",
        default="bpm23",
        type=str,
        help="Wandb project name",
    )

    return parser


def main(params=None):
    # params = get_args_parser().parse_args()
    if params.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Warning; gpu not available") #todo

    logger = wandb.init(config=params, project=params.project_name)
    # todo: how to diff sweep from single/manual run
    # logger.config.update(
    #     {
    #         "dataset": params.dataset,
    #         "condition": params.condition,
    #         "device": device.type,
    #     },
    # )
    config = logger.config
    log = read_data(f"data/{params.dataset}/log.csv")
    log["target"] = log[params.condition]
    log.drop(["trace_time", "resource_usage", "variant"], axis=1, inplace=True)
    log = prepare_log(log)
    vocabs = get_vocabs(log)
    # encoding

    for f in vocabs:
        log.loc[:, f] = log.loc[:, f].transform(lambda x: vocabs[f]["stoi"][x])

    if "resource" not in vocabs:  # i.e. resource is numerical
        # z score normalization for numerical resource
        if params.condition == "resource_usage":
            # the res usage condition makes no sense for numerical resource
            logger.finish()
            return
        mean = log.loc[log.type_set == "train", "resource"].mean()
        std = log.loc[log.type_set == "train", "resource"].std()
        log.loc[:, "resource"] = (log.loc[:, "resource"] - mean) / std

    # ToDo how to track cat features? here we have (act, res, rt)
    data_train, data_test = vectorize_log(log)

    train_loader = get_loader(data_train, batch_size=config.batch_size)
    test_loader = get_loader(data_test, batch_size=1024, shuffle=False)

    torch.manual_seed(0)
    # model = MTCondLSTM(vocabs=vocabs, batch_size=config.batch_size)
    model = MTCondDG(vocabs=vocabs, batch_size=config.batch_size)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0.5)

    model.apply(init_weights)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.to(device)
    wandb.watch(model, log="all")
    # X, y = next(iter(train_loader))
    # X = [x.to(device) for x in X]
    # model(X)

    if "resource" in vocabs:
        res_loss = nn.CrossEntropyLoss()
    else:
        res_loss = nn.MSELoss()
    criterion = {"clf": nn.CrossEntropyLoss(), "res": res_loss, "reg": nn.MSELoss()}
    if config.optimizer == "sgd":
        optm = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optm = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    sc = MultiStepLR(optm, milestones=[25, 35], gamma=0.1)
    pprint.pprint(config)
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
    main(params)
