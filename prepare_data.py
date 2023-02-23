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


def log_exists(args):
    """Check if a preprocessed log already exists.

    Args:
        args (_type_): arguments
    """
    output_path = os.path.join("data", args.dataset)
    ensure_dir(output_path)
    if args.overwrite:
        clean_directory(output_path)
        return False
    else:
        return os.path.exists(os.path.join(output_path, "log.csv"))


LABEL_FUNCTION = {"trace_time": pp.trace_time, "resource_usage": pp.resource_usage}
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    output = os.path.join("data", args.dataset)

    if not log_exists(args):
        train = read_data(os.path.join(args.path, "train.csv"), format_cols=True)
        test = read_data(os.path.join(args.path, "test.csv"), format_cols=True)
        train["type_set"] = "train"
        test["type_set"] = "test"
        df = pd.concat((train, test))
        df = create_features(df)
    else:
        df = pd.read_csv(os.path.join(output, "log.csv"))

    if args.condition in df.columns:
        pass
    else:
        df[args.condition] = LABEL_FUNCTION[args.condition](df)
        df.to_csv(os.path.join(output, "log.csv"), index=False)
