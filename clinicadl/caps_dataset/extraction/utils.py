import errno
import json
from copy import copy
from pathlib import Path
from typing import Any, Dict


def change_path_to_str(obj: dict, key: str, value) -> dict:
    if (
        key.endswith("tsv")
        or key.endswith("dir")
        or key.endswith("directory")
        or key.endswith("path")
        or key.endswith("json")
        or key.endswith("location")
    ):
        if not value:
            obj[key] = ""
        elif isinstance(value, Path):
            obj[key] = value.as_posix()

    return obj


def change_str_to_path(obj: dict, key: str, value):
    if (
        key.endswith("tsv")
        or key.endswith("dir")
        or key.endswith("directory")
        or key.endswith("path")
        or key.endswith("json")
        or key.endswith("location")
    ):
        if value == "" or value is False or value is None:
            obj[key] = False
        else:
            obj[key] = Path(value)

    return obj


def path_encoder(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    change_path_to_str(obj, key2, value2)
            else:
                change_path_to_str(obj, key, value)
    return obj


def path_decoder(obj):
    if isinstance(obj, dict):
        obj2 = copy(obj)
        for key, value in obj2.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    change_str_to_path(obj, key2, value2)
            else:
                change_str_to_path(obj, key, value)


def write_preprocessing(
    preprocessing_dict: Dict[str, Any], caps_directory: Path
) -> Path:
    extract_dir = caps_directory / "tensor_extraction"
    extract_dir.mkdir(parents=True, exist_ok=True)

    json_path = extract_dir / preprocessing_dict["extract_json"]

    if json_path.is_file():
        raise FileExistsError(
            f"JSON file at {json_path} already exists. "
            f"Please choose another name for your preprocessing file."
        )

    with json_path.open(mode="w") as json_file:
        json.dump(preprocessing_dict, json_file, default=path_encoder)
    return json_path


def read_preprocessing(json_path: Path) -> Dict[str, Any]:
    if json_path.suffix != ".json":
        json_path = json_path.with_suffix(".json")

    if not json_path.is_file():
        raise FileNotFoundError(errno.ENOENT, json_path)

    try:
        with json_path.open(mode="r") as f:
            preprocessing_dict = json.load(f)
    except IOError as e:
        raise IOError(f"Error reading json preprocessing file {json_path}: {e}")
    return preprocessing_dict
