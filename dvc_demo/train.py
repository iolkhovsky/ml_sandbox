import argparse
from os import makedirs
from os.path import isdir, join
import pandas as pd
import pickle
import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
import sklearn.neural_network
from time import time

from utils import read_yaml, save_object, visualize_classification_2d, write_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default="training.yml",
                        help="Absolute path to the training pipeline")
    return parser.parse_args()


def run_training(args):
    config = read_yaml(args.config)
    report = {
        "config": config
    }

    train_df = pd.read_csv(config["dataset"]["train"])
    x_train = train_df.loc[:, train_df.columns != "label"].to_numpy()
    y_train = train_df.loc[:, train_df.columns == "label"].to_numpy()
    test_df = pd.read_csv(config["dataset"]["test"])
    x_test = test_df.loc[:, test_df.columns != "label"].to_numpy()
    y_test = test_df.loc[:, test_df.columns == "label"].to_numpy()

    preprocessor = getattr(sklearn.preprocessing, config["preprocessor"]["type"])()
    preprocessor.fit(x_train)
    x_train_norm = preprocessor.transform(x_train)
    x_test_norm = preprocessor.transform(x_test)

    model_pars = config["model"]["parameters"]
    if model_pars != "None":
        model = getattr(getattr(sklearn, config["model"]["method"]), config["model"]["type"])(**model_pars)
    else:
        model = getattr(getattr(sklearn, config["model"]["method"]), config["model"]["type"])()

    start = time()
    model.fit(x_train_norm, y_train)
    training_time = time() - start
    report["training_time"] = training_time

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
    save_object(model, join(config["output"], "model.pickle"))
    save_object(preprocessor, join(config["output"], "preprocessor.pickle"))
    write_yaml(join(config["output"], "report.yml"), report)
    feature_names = train_df.columns[:-1].values.tolist()
    visualize_classification_2d(x_train_norm, y_train.flatten(), model, path=join(config["output"], "train.png"),
                                hint="Train subset", feature_names=feature_names)
    visualize_classification_2d(x_test_norm, y_test.flatten(), model, path=join(config["output"], "test.png"),
                                hint="Test subset", feature_names=feature_names)


if __name__ == "__main__":
    run_training(parse_args())
