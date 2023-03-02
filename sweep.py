import wandb
from train import main


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
        "--project-name",
        default="test",
        type=str,
        help="Wandb project name",
    )

    return parser


if __name__ == "__main__":
    params = get_args_parser().parse_args()
    sweep_config = {
        "method": "bayes",
        "name": f"{params.dataset}-{params.condition}",
        "metric": {"name": "test_loss", "goal": "minimize"},
        "parameters": {
            "optimizer": {"values": ["adam", "sgd"]},
            "lr": {"max": 1e-3, "min": 1e-6},
            "epochs": {"values": [50]},
            "batch_size": {"values": [64, 256, 512]},
            "weight_decay": {"values": [0.0, 1e-2, 1e-3]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project=params.project_name)
    wandb.agent(sweep_id, main, count=1)
