import errno
import json
import os
from typing import Any, Dict


def write_preprocessing(preprocessing_dict: Dict[str, Any], caps_directory: str):
    extract_dir = os.path.join(
        caps_directory,
        "tensor_extraction",
    )
    os.makedirs(extract_dir, exist_ok=True)
    json_path = os.path.join(extract_dir, preprocessing_dict["json_name"])
    if os.path.exists(json_path):
        raise FileExistsError(
            f"JSON file at {json_path} already exists. "
            f"Please choose another name for your preprocessing file."
        )

    with open(json_path, "w") as json_file:
        json.dump(preprocessing_dict, json_file, indent=2)
    return json_path


def read_preprocessing(json_path: str) -> Dict[str, Any]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), json_path)
    try:
        with open(json_path, "r") as f:
            preprocessing_dict = json.load(f)
    except IOError:
        raise IOError(f"Cannot open json preprocessing file {json_path}")
    return preprocessing_dict
