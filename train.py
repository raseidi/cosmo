import pprint
import warnings
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cosmo.event_logs import ConstrainedContinuousTraces
from cosmo.event_logs.utils import collate_fn

# from cosmo.models import NeuralNet
from cosmo.engine import train
from cosmo.event_logs import get_declare, LOG_READERS
import argparse
from cosmo.models import Cosmo
from cosmo.utils import experiment_exists

try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False

# seed everything
torch.manual_seed(42)


def read_args():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="sepsis")
    args.add_argument("--lr", type=float, default=5e-4)
    args.add_argument("--batch-size", type=int, default=32)
    args.add_argument("--weight-decay", type=float, default=1e-5)
    args.add_argument("--epochs", type=int, default=1000)
    args.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args.add_argument("--shuffle-dataset", type=bool, default=True)
    args.add_argument("--hidden-size", type=int, default=32)
    args.add_argument("--input-size", type=int, default=8)
    args.add_argument("--project-name", type=str, default="cosmo-bpm-sim")
    args.add_argument("--grad-clip", type=float, default=None)
    args.add_argument("--n-layers", type=int, default=1)
    args.add_argument("--wandb", type=str, default="False")
    args.add_argument("--template", type=str, default="all")
    args.add_argument("--backbone", type=str, default="crnn")
    args.add_argument("--lora", type=str, default="False")
    args.add_argument("--r-rank", type=int, default=32)
    args.add_argument("--lora-alpha", type=int, default=64)
    args.add_argument("--n-heads", type=int, default=1)
    args = args.parse_args()

    args.lora = args.lora == "True"
    args.wandb = args.wandb == "True"
    return args


def run(config):
    log_reader = LOG_READERS.get(config["dataset"], None)
    if log_reader is None:
        raise ValueError(f"Dataset {config['dataset']} not found")
    log = log_reader()

    declare_constraints = get_declare(config["dataset"], templates=config["template"])

    not_found_constraints = set(log.case_id.unique()) - set(
        declare_constraints.case_id.unique()
    )
    if not_found_constraints:
        warnings.warn(
            f"Dropping constraints not found for {len(not_found_constraints)} case(s)"
        )
        log = log[~log.case_id.isin(not_found_constraints)]

    train_set, test_set = log[log["split"] == "train"], log[log["split"] == "test"]
    train_dataset = ConstrainedContinuousTraces(
        log=train_set,
        constraints=declare_constraints.copy(),
        continuous_features=["remaining_time_norm"],
        categorical_features=["activity"],
        dataset_name=config["dataset"] + "_" + config["template"],
        train=True,
        device=config["device"],
    )
    config["n_features"] = train_dataset.num_features
    test_dataset = ConstrainedContinuousTraces(
        log=test_set,
        vocab=train_dataset.get_vocabs(),
        constraints=declare_constraints.copy(),
        continuous_features=["remaining_time_norm"],
        categorical_features=["activity"],
        dataset_name=config["dataset"] + "_" + config["template"],
        train=False,
        device=config["device"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = Cosmo(
        vocabs=train_dataset.feature2idx,
        n_continuous=train_dataset.num_cont_features,
        n_constraints=train_dataset.num_constraints,
        backbone_model=config["backbone"],
        embedding_size=config["input_size"],
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
        lora=True,
        r_rank=config["r_rank"],
        lora_alpha=config["lora_alpha"],
        n_heads=config["n_heads"],
    )
    model.to(config["device"])

    optim = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim, T_max=config["epochs"] * config["batch_size"], verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()

    # run_name = f"backbone={config['backbone']}-templates={config['template']}-lr={config['lr']}-bs={config['batch_size']}-hidden={config['hidden_size']}-input={config['input_size']}-nlayers={config['n_layers']}-rank={config['r_rank']}-alpha={config['lora_alpha']}-lora={str(config['lora'])}"
    run_name = f"backbone={config['backbone']}-lr={config['lr']}-bs={config['batch_size']}-hidden={config['hidden_size']}-input={config['input_size']}-nheads={config['n_heads']}"

    if config["wandb"] and _has_wandb:
        wandb.init(project=config["project_name"], config=config, name=run_name)
        wandb.watch(model, log="all")

    config["run_name"] = run_name

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optim,
        scaler=scaler,
        config=config,
        scheduler=scheduler,
    )
    if config["wandb"] and _has_wandb:
        wandb.finish()


if __name__ == "__main__":
    config = read_args()
    config = vars(config)

    print("\n\nConfig:")
    # pprint.pprint(config)

    if experiment_exists(config):
        print("Experiment exists, skipping...\n\n")
        exit(0)

    # # temporary loop to persist selected models
    # import pandas as pd
    # selected_models = pd.read_csv("selected_models.csv")
    # COLS = ["dataset", "backbone", "lr", "epochs", "batch_size", "template", "n_layers", "input_size", "hidden_size", ]
    # for ix, row in selected_models.iterrows():
    #     if ix <= 14:
    #         continue
    #     for c in COLS:
    #         config[c] = row[c]

    #     if row.backbone == "vanilla":
    #         config["template"] = "all"
    #     pprint.pprint(config)
    run(config)

    # import cProfile
    # cProfile.runctx('run(config)', globals(), locals(), filename="train.prof", sort="cumtime")
