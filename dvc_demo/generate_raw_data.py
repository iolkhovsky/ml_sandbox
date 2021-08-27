import argparse
import numpy as np
from os import makedirs
from os.path import isdir, join
import pandas as pd
from sklearn.datasets import make_classification

from utils import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Toy generator of a raw dataset")
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
    data = np.hstack([X, np.expand_dims(y, axis=1)])
    columns = [f"feature_{i}" for i in range(config["dataset"]["n_features"])] + ["label"]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(join(dataset_path, "raw_data.csv"), index=False)


if __name__ == "__main__":
    generate_data(parse_args())
