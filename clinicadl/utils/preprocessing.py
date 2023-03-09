import errno
import json
from pathlib import Path
from typing import Any, Dict


def write_preprocessing(preprocessing_dict: Dict[str, Any], caps_directory: str):
    extract_dir = Path(caps_directory) / "tensor_extraction"
    extract_dir.mkdir(parents=True, exist_ok=True)
    json_path = extract_dir / preprocessing_dict["extract_json"]
    if json_path.is_file():
        raise FileExistsError(
            f"JSON file at {json_path} already exists. "
            f"Please choose another name for your preprocessing file."
        )

    with open(json_path, "w") as json_file:
        json.dump(preprocessing_dict, json_file, indent=2)
    return json_path


def read_preprocessing(json_path: str) -> Dict[str, Any]:
    if not json_path.name.endswith(".json"):
        json_path += ".json"

    if not json_path.is_file():
        raise FileNotFoundError(errno.ENOENT, json_path)
    try:
        with open(json_path, "r") as f:
            preprocessing_dict = json.load(f)
    except IOError:
        raise IOError(f"Cannot open json preprocessing file {json_path}")
    return preprocessing_dict
