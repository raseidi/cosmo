import os, sys
import numpy as np
import pandas as pd
from generator import preprocessing as pp
from generator.utils import ensure_dir, read_data


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Data preprocessing for Conditioned Simulator",
        add_help=add_help,
    )
    parser.add_argument(
        "--path",
        default="/home/seidi/datasets/logs/bpi20/RequestForPayment/train_test",
        type=str,
        help="Path for benchmarked datasets (Weytjens and Weerdt, 2022)",
    )
    parser.add_argument(
        "--condition",
        default="trace_time",
        type=str,
        help="Condition for labeling cases.",
    )
    parser.add_argument(
        "--dataset",
        default="RequestForPayment",
        type=str,
        help="Dataset name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If an existing directory should be overwritten.",
    )

    return parser


def create_features(df):
    # df["tf1"] = df.groupby("case_id")["time"].transform(execution_time)
    # df["tf2"] = df.groupby("case_id")["time"].transform(accumulated_time)
    # df["tf3"] = df["time"].transform(within_day)
    # df["tf4"] = df["time"].transform(within_week)
    df["remaining_time"] = df.groupby("case_id")["time"].transform(pp.remaining_time)
    df.loc[:, ["remaining_time"]] = (
        df.loc[:, ["remaining_time"]]
        .apply(lambda x: x.dt.total_seconds())
        .apply(np.log1p)
    )  # / (24 * 60 * 60) # remaining time in days
    return df


def clean_directory(path):
    import shutil

    shutil.rmtree(path)


def check_directory(args):
    output_path = os.path.join("data", args.dataset, args.condition)
    ensure_dir(output_path)
    if args.overwrite:
        clean_directory(output_path)
    else:
        if os.path.exists(os.path.join(output_path, "log.csv")):
            import sys
            sys.exit(1)
            # raise Exception(
            #     "A log.csv already exists for this dataset under the provided condition. Set -o argument as True if you wish to overwrite it."
            # )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    check_directory(args)

    train = read_data(os.path.join(args.path, "train.csv"), format_cols=True)
    test = read_data(os.path.join(args.path, "test.csv"), format_cols=True)
    train["type_set"] = "train"
    test["type_set"] = "test"
    df = pd.concat((train, test))
    df = create_features(df)

    # condition
    df["target"] = pp.trace_time(df)

    save_path = os.path.join("data", args.dataset, args.condition)
    df.to_csv(os.path.join(save_path, "log.csv"), index=False)
