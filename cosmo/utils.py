import os
import pandas as pd


def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def experiment_exists(config: dict):
    """Check if experiment with the same config exists."""

    experiments = get_existing_experiments()
    if experiments is None:
        return False

    # TODO: make this drop generic
    experiments.drop(columns=["id", "n_features"], inplace=True, errors="ignore")
    # experiments.fillna(value=None, axis=0, inplace=True)  # none is not valid
    # grad_clip is None for config and nan for exp
    for _, exp in experiments.T.to_dict().items():
        exp["grad_clip"] = None
        if exp == config:
            return True
    return False


def get_existing_experiments(force_fetch=False, project="cosmo-ltl"):
    if os.path.exists("experiments.csv") and not force_fetch:
        try:
            return pd.read_csv("experiments.csv")
        except:
            return None
    else:
        return fetch_experiments(project)


def fetch_experiments(project="cosmo-v4"):
    try:
        import wandb
    except ImportError:
        print("wandb not installed")
        return

    api = wandb.Api()
    runs = api.runs("raseidi/" + project)
    metrics = [
        "train_a_loss",
        "train_a_acc",
        "train_t_loss",
        "test_a_loss",
        "test_a_acc",
        "test_t_loss",
        "_runtime",
    ]

    experiments = pd.DataFrame()
    for r in runs:
        if r.state != "finished":
            continue

        new = pd.DataFrame([r.config])
        new["id"] = r.id
        new["name"] = r.name

        for m in metrics:
            new[m] = r.summary[m]
        experiments = pd.concat((experiments, new), ignore_index=True)

    experiments.reset_index(inplace=True, drop=True)
    experiments.to_csv("experiments.csv", index=False)
    return experiments
