import argparse
from os import makedirs
from os.path import isdir, join
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from time import time

from utils import read_yaml, read_object, visualize_classification_2d, write_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--config", type=str, default="testing.yml",
                        help="Absolute path to the testing config")
    return parser.parse_args()


def run_testing(args):
    config = read_yaml(args.config)
    report = {
        "config": config
    }

    test_df = pd.read_csv(config["dataset"])
    x_test = test_df.loc[:, test_df.columns != "label"].to_numpy()
    y_test = test_df.loc[:, test_df.columns == "label"].to_numpy()

    preprocessor = read_object(join(config["model"], "preprocessor.pickle"))
    model = read_object(join(config["model"], "model.pickle"))
    x_test_norm = preprocessor.transform(x_test)

    start = time()
    y_pred = model.predict(x_test_norm)
    prediction_time = time() - start
    report["prediction_time"] = prediction_time

    report["quality"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    if not isdir(config["output"]):
        makedirs(config["output"])
    write_yaml(join(config["output"], "test_report.yml"), report)
    feature_names = test_df.columns[:-1].values.tolist()
    visualize_classification_2d(x_test_norm, y_test.flatten(), model, path=join(config["output"], "test.png"),
                                hint="Test subset", feature_names=feature_names)


if __name__ == "__main__":
    run_testing(parse_args())
