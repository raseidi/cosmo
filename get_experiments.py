import argparse
from cosmo.utils import fetch_experiments


def read_args():
    args = argparse.ArgumentParser()
    args.add_argument("--project", type=str, default="raseidi/cosmo-v4")
    return args.parse_args()


if __name__ == "__main__":
    args = read_args()
    _ = fetch_experiments(project=args.project)
