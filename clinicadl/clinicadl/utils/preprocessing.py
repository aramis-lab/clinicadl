import json
import errno
import os


def write_preprocessing(preprocessing_dict, json_path):
    with open(json_path, "w") as json_file:
        json.dump(preprocessing_dict, json_file)


def read_preprocessing(json_path):
    if not os.path.isfile(json_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), json_path)
    try:
        with open(json_path, "r") as f:
            preprocessing_dict = json.load(f)
    except:
        raise IOError(f"cannot open json preprocessing file {json_path}")
    return preprocessing_dict
