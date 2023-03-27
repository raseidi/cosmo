import os
import wandb
import numpy as np
import pandas as pd
from cosmo import preprocessing as pp
from cosmo.utils import ensure_dir, read_data


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
    )
    return df


def preprocess(df):
    """Preprocessing event logs

    1. Exclude cases with len(trace) < 3
    2. Exclude unfrequent variants
    3. Fill nans of numerical values with mean from case_id
    4. Fill nans of categorical values with most frequent from case_id (ToDo)

    Args:
        df (pd.DataFrame): Event log

    Returns:
        pd.DataFrame: Preprocessed event log
    """

    def drop_unfrequent(df, attribute="case_id", threshold=3):
        ix = df[attribute].value_counts() >= threshold
        ix = ix[ix == True].index
        df = df[df[attribute].isin(ix)]
        return df

    df = drop_unfrequent(df, attribute="case_id")
    df = pp.label_variants(df)
    df = drop_unfrequent(df, attribute="variant")
    exclude_cols = ["case_id", "time", "remaining_time", "type_set", "target"]
    numerical = df.select_dtypes(include=["number"]).columns.difference(exclude_cols)

    # filling missing values
    df.reset_index(inplace=True, drop=True)
    for n in numerical:
        if df[n].isna().sum():
            df.loc[:, n] = df.groupby("case_id")[n].transform(
                lambda x: x.fillna(x.mean())
            )
        if (
            df[n].isna().sum()
        ):  # if the whole trace doesn't have resource, the nan won't be filled at the previous step
            df.loc[:, n] = df.loc[:, n].fillna(lambda x: x.mean())
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
        print("[!] Overwriting")
        clean_directory(output_path)
        ensure_dir(output_path)
        return False
    else:
        return os.path.exists(os.path.join(output_path, "log.csv"))


LABEL_FUNCTION = {"trace_time": pp.trace_time, "resource_usage": pp.resource_usage}
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    output = os.path.join("data", args.dataset)
    output_file = os.path.join(output, "log.csv")
    if not log_exists(args):
        train = read_data(os.path.join(args.path, "train.csv"), format_cols=True)
        test = read_data(os.path.join(args.path, "test.csv"), format_cols=True)
        train["type_set"] = "train"
        test["type_set"] = "test"
        df = pd.concat((train, test))
        df = create_features(df)
        df = preprocess(df)
        if df.isna().sum().any():
            print("-" * 80)
    else:
        df = pd.read_csv(output_file)

    for condition in LABEL_FUNCTION:
        if condition not in df.columns:
            df[condition] = LABEL_FUNCTION[condition](df)

    df.to_csv(output_file, index=False)

    run = wandb.init(project="bpm23", job_type="dataset-creation")
    dataset = wandb.Artifact("RequestForPayment", type="dataset")
    dataset.add_file("data/RequestForPayment/log.csv")
    run.log_artifact(dataset)

    wandb.finish()
