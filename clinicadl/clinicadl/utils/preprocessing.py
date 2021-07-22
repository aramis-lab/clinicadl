import json


def write_preprocessing(preprocessing_dict, json_path):
    with open(json_path, "w") as json_file:
        json.dump(preprocessing_dict, json_file)


def read_preprocessing(json_path):
    with open(json_path, "r") as f:
        preprocessing_dict = json.load(f)

    return preprocessing_dict
