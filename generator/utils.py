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


def get_runs(run_name="trace-generation-time-condition"):
    import wandb
    import pandas as pd

    api = wandb.Api()

    runs = api.runs(run_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    summary = pd.DataFrame(summary_list, index=range(len(runs)))
    configs = pd.DataFrame(config_list, index=range(len(runs)))
    names = pd.DataFrame(name_list, index=range(len(runs)), columns=["run_name"])
    df = pd.concat((names, summary, configs), axis=1)
    return df


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
        df = df.loc[
            :,
            [
                "case:concept:name",
                "concept:name",
                "org:resource",
                "time:timestamp",
            ],
        ]
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


def get_runs(run_name="trace-generation-time-condition"):
    import wandb
    import pandas as pd

    api = wandb.Api()

    runs = api.runs(run_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    summary = pd.DataFrame(summary_list, index=range(len(runs)))
    configs = pd.DataFrame(config_list, index=range(len(runs)))
    names = pd.DataFrame(name_list, index=range(len(runs)), columns=["run_name"])
    df = pd.concat((names, summary, configs), axis=1)
    return df
