import argparse

from utils import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default="training.yml",
                        help="Absolute path to the training pipeline")
    return parser.parse_args()


def run_training(args):
    config = read_yaml(args.config)


if __name__ == "__main__":
    run_training(parse_args())
