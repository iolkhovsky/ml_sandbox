import argparse
import numpy as np
from os import makedirs
from os.path import isdir, join
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from utils import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Eye patches generator")
    parser.add_argument("--config", type=str, default="data_generator.yml",
                        help="Absolute path to source videofile")
    return parser.parse_args()


def generate_data(args):
    config = read_yaml(args.config)
    dataset_path = config["output"]
    if not isdir(dataset_path):
        makedirs(dataset_path)
    dataset_pars = config["dataset"]
    X, y = make_classification(**dataset_pars)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=config["split"]["test_share"], random_state=config["split"]["random_state"]
    )
    train_data = np.hstack([x_train, np.expand_dims(y_train, axis=1)])
    test_data = np.hstack([x_test, np.expand_dims(y_test, axis=1)])
    columns = [f"feature_{i}" for i in range(config["dataset"]["n_features"])] + ["label"]
    train_df = pd.DataFrame(data=train_data, columns=columns)
    test_df = pd.DataFrame(data=test_data, columns=columns)
    train_df.to_csv(join(dataset_path, "train.csv"))
    test_df.to_csv(join(dataset_path, "test.csv"))


if __name__ == "__main__":
    generate_data(parse_args())
