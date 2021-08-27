from os.path import dirname, isfile, isdir
from os import makedirs
import pickle
import yaml


def read_yaml(path):
    assert isfile(path), f"{path} doesnt exist"
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        return data


def write_yaml(path, data):
    parent_folder = dirname(path)
    if len(parent_folder) and not isdir(parent_folder):
        makedirs(parent_folder)
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)


def save_object(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_object(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
        return obj
