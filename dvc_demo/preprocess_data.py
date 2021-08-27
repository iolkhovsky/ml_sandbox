import argparse
import numpy as np
from os import makedirs
from os.path import isdir, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--data", type=str, default="data/raw_data.csv",
                        help="Absolute path to the source dataset")
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--test_share", type=float, default=0.2,
                        help="Test subset ratio")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state")
    return parser.parse_args()


def preprocess_data(args):
    if not isdir(args.output):
        makedirs(args.output)

    df = pd.read_csv(args.data, index_col=False)
    features = df.loc[:, df.columns != "label"].to_numpy()
    labels = df.loc[:, df.columns == "label"].to_numpy()

    norm_features = StandardScaler().fit_transform(features, labels)
    feature_selector = SelectKBest(f_classif, k=2)
    feature_selector.fit(norm_features, labels)
    best_features_mask = feature_selector.get_support()
    best_features_labels = df.columns[:-1][best_features_mask].values.tolist()

    best_features = df[best_features_labels].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        best_features, labels, test_size=args.test_share, random_state=args.random_state
    )

    train_df = pd.DataFrame(data=np.hstack([x_train, y_train]), columns=best_features_labels + ["label"])
    train_df.to_csv(join(args.output, "train.csv"), index=False)
    test_df = pd.DataFrame(data=np.hstack([x_test, y_test]), columns=best_features_labels + ["label"])
    test_df.to_csv(join(args.output, "test.csv"), index=False)


if __name__ == "__main__":
    preprocess_data(parse_args())
