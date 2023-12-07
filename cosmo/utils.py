import os
import torch
import shutil


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, "latest_checkpoint.txt")

    ckpt_name = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [
                ckpt_name + " acc: " + f'{state["test_acc"]:.4f}' + "\n"
            ] + ckpt_list
    else:
        ckpt_list = [ckpt_name + " acc: " + f'{state["test_acc"]:.4f}' + "\n"]

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, "w") as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, "best_model.ckpt"))


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, "best_model.ckpt")
        else:
            with open(os.path.join(ckpt_dir_or_file, "latest_checkpoint.txt")) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(" [*] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_vocabs(log, categorical_features=None):
    if categorical_features is None:
        exclude_cols = ["case_id", "time", "remaining_time", "type_set", "target"]
        categorical_features = log.select_dtypes(include=["object"]).columns.difference(
            exclude_cols
        )
    vocabs = dict()
    for f in categorical_features:
        stoi = {v: k for k, v in enumerate(log.loc[:, f].unique())}
        vocabs[f] = {
            "stoi": stoi,
            "size": len(stoi),
            "emb_dim": int(len(stoi) ** (1 / 2))
            if int(len(stoi) ** (1 / 2)) > 2
            else 2,
        }
    return vocabs


def read_data(path, format_cols=False):
    """Read benchmarked event logs.

    Running the repository provided by Weytjens and Weerdt (2022) results in
    two event logs (train.csv and test.csv). ``path'' indicates the folder
    containing both datasets.
    Args:
        path (str): path for directory containing splitted event logs

    Returns:
        pd.DataFrame: a pd.DataFrame containing the concatened train and test sets
    """
    import pandas as pd

    df = pd.read_csv(path)
    if format_cols:
        # df = df.loc[
        #     :,
        #     [
        #         "case:concept:name",
        #         "concept:name",
        #         "org:resource",
        #         "time:timestamp",
        #     ],
        # ]
        df = df.rename(
            columns={
                "case:concept:name": "case_id",
                "concept:name": "activity",
                "org:resource": "resource",
                "time:timestamp": "time",
            }
        )
    df["time"] = pd.to_datetime(df["time"], infer_datetime_format=True).dt.tz_localize(
        None
    )
    return df


def get_runs(
    project="multi-task",
    best_epoch=True,
    best_metric="test_loss",
):
    import wandb
    import pandas as pd

    api = wandb.Api()
    runs = api.runs(f"raseidi/{project}")

    cols = None
    results = pd.DataFrame()
    for r in runs:
        if r.state != "finished":
            continue

        hist = r.history()
        if hist.empty:
            continue
        if cols is None:
            cols = [
                c
                for c in r.history().columns
                if not c.startswith("param") and not c.startswith("grad")
            ]

        hist = hist[cols]
        hist.loc[:, "run_name"] = r.name
        if best_epoch:
            if "loss" in best_metric:
                best_step = hist[best_metric].idxmin()
            else:
                best_step = hist[best_metric].idxmax()
            hist = hist.loc[hist._step == best_step, :]
            hist = hist.join(pd.DataFrame([r.config], index=[best_step]))

        results = pd.concat((results, hist.iloc[[-1], :]))

    results.reset_index(inplace=True, drop=True)
    return results
