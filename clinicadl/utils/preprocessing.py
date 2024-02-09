import errno
import json
from pathlib import Path
from typing import Any, Dict

from clinicadl.utils.maps_manager.maps_manager_utils import change_path_to_str


def write_preprocessing(preprocessing_dict: Dict[str, Any], caps_directory: Path):
    extract_dir = caps_directory / "tensor_extraction"
    extract_dir.mkdir(parents=True, exist_ok=True)
    json_path = extract_dir / preprocessing_dict["extract_json"]
    if json_path.is_file():
        raise FileExistsError(
            f"JSON file at {json_path} already exists. "
            f"Please choose another name for your preprocessing file."
        )
    preprocessing_dict_bis = change_path_to_str(preprocessing_dict)
    with json_path.open(mode="w") as json_file:
        json.dump(preprocessing_dict_bis, json_file)
    return json_path


def read_preprocessing(json_path: Path) -> Dict[str, Any]:
    if not json_path.name.endswith(".json"):
        json_path = json_path.with_suffix(".json")

    if not json_path.is_file():
        raise FileNotFoundError(errno.ENOENT, json_path)
    try:
        with json_path.open(mode="r") as f:
            preprocessing_dict = json.load(f)
    except IOError:
        raise IOError(f"Cannot open json preprocessing file {json_path}")
    return preprocessing_dict
