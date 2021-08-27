import argparse
import numpy as np
from os import makedirs
from os.path import isdir, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from utils import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--config", type=str, default="params.yaml",
                        help="Absolute path to the configuration file (params.yaml)")
    return parser.parse_args()


def preprocess_data(args):
    config = read_yaml(args.config)["preprocessing"]
    output = config["output"]
    data = config["data"]
    test_share = config["test_share"]
    random_state = config["random_state"]

    if not isdir(output):
        makedirs(output)

    df = pd.read_csv(data, index_col=False)
    features = df.loc[:, df.columns != "label"].to_numpy()
    labels = df.loc[:, df.columns == "label"].to_numpy()

    norm_features = StandardScaler().fit_transform(features, labels)
    feature_selector = SelectKBest(f_classif, k=2)
    feature_selector.fit(norm_features, labels)
    best_features_mask = feature_selector.get_support()
    best_features_labels = df.columns[:-1][best_features_mask].values.tolist()

    best_features = df[best_features_labels].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        best_features, labels, test_size=test_share, random_state=random_state
    )

    train_df = pd.DataFrame(data=np.hstack([x_train, y_train]), columns=best_features_labels + ["label"])
    train_df.to_csv(join(output, "train.csv"), index=False)
    test_df = pd.DataFrame(data=np.hstack([x_test, y_test]), columns=best_features_labels + ["label"])
    test_df.to_csv(join(output, "test.csv"), index=False)


if __name__ == "__main__":
    preprocess_data(parse_args())
