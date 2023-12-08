import os
import torch
import shutil

from cosmo.utils import ensure_dir


def save_checkpoint(checkpoint: dict, experiment_id: str | int, config: dict):
    save_path = os.path.join("models", config["dataset"], experiment_id + ".pth")
    ensure_dir(os.path.dirname(save_path))

    torch.save(checkpoint, save_path)


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
