import pprint
import warnings
import wandb
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cosmo.event_logs import ConstrainedContinuousTraces
from cosmo.event_logs.utils import collate_fn
from cosmo.models import NeuralNet
from cosmo.engine import train
from cosmo.event_logs import get_declare, LOG_READERS
import argparse
from cosmo.utils import experiment_exists

# seed everything
torch.manual_seed(42)

def read_args():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="bpi20_permit")
    args.add_argument("--lr", type=float, default=0.0005)
    args.add_argument("--batch-size", type=int, default=16)
    args.add_argument("--weight-decay", type=float, default=1e-1)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args.add_argument("--shuffle-dataset", type=bool, default=True)
    args.add_argument("--hidden-size", type=int, default=256)
    args.add_argument("--input-size", type=int, default=32)
    args.add_argument("--project-name", type=str, default="cosmo-v7")
    args.add_argument("--grad-clip", type=float, default=None)
    args.add_argument("--n-layers", type=int, default=1)
    args.add_argument("--wandb", type=bool, default=False)
    args.add_argument("--template", type=str, default="all")

    return args.parse_args()


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

    # model
    model = NeuralNet(
        vocabs=train_dataset.feature2idx,
        continuous_size=train_dataset.num_cont_features,
        constraint_size=train_dataset.num_constraints,
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    # optim = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=10, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim, T_max=config["epochs"] * config["batch_size"], verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()


    run_name = f"{config['dataset']}-templates={config['template']}-n_features={train_dataset.num_features}-lr={config['lr']}-bs={config['batch_size']}-wd={config['weight_decay']}-epochs{config['epochs']}-hidden={config['hidden_size']}-input={config['input_size']}-gradclip={config['grad_clip']}-nlayers={config['n_layers']}"
    
    if config["wandb"]:
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
    if config["wandb"]:
        wandb.finish()

if __name__ == "__main__":
    config = read_args()
    config = vars(config)
    if config["dataset"] == "bpi19":
        exit(0)
    
    print("\n\nConfig:")
    pprint.pprint(config)

    if experiment_exists(config):
        print("Experiment exists, skipping...\n\n")
        exit(0)

    if config["hidden_size"] < config["input_size"]:
        print("Hidden size must be greater than input size")
        exit(1)

    print("Running...")
    run(config)
    # import cProfile
    # cProfile.runctx('run(config)', globals(), locals(), filename="train.prof", sort="cumtime")

