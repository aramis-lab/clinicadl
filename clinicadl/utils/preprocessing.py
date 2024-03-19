import errno
import json
from pathlib import Path
from typing import Any, Dict


def path_encoder(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if (
                        key2.endswith("tsv")
                        or key2.endswith("dir")
                        or key2.endswith("directory")
                        or key2.endswith("path")
                        or key2.endswith("json")
                        or key2.endswith("location")
                    ):
                        if not value2:
                            obj[value][key2] = ""
                        elif isinstance(value2, Path):
                            obj[value][key2] = value2.as_posix()
            else:
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


def path_decoder(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if (
                        key2.endswith("tsv")
                        or key2.endswith("dir")
                        or key2.endswith("directory")
                        or key2.endswith("path")
                        or key2.endswith("json")
                        or key2.endswith("location")
                    ):
                        if value2 == "" or value2 is False:
                            obj[key][key2] = False
                        else:
                            obj[key][key2] = Path(value2)
            else:
                if (
                    key.endswith("tsv")
                    or key.endswith("dir")
                    or key.endswith("directory")
                    or key.endswith("path")
                    or key.endswith("json")
                    or key.endswith("location")
                ):
                    if value == "" or value is False:
                        obj[key] = False
                    else:
                        obj[key] = Path(value)
    return obj


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
    if not json_path.suffix == ".json":
        json_path = json_path.with_suffix(".json")

    if not json_path.is_file():
        raise FileNotFoundError(errno.ENOENT, json_path)

    try:
        with json_path.open(mode="r") as f:
            preprocessing_dict = json.load(f)
    except IOError as e:
        raise IOError(f"Error reading json preprocessing file {json_path}: {e}")
    return preprocessing_dict
