import enum
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import torch

from clinicadl.utils.typing import PathType


def read_json(json_path: PathType) -> Dict[str, Any]:
    """
    Reads the serialized config class from a JSON file.
    """
    json_path = Path(json_path)

    if not json_path.is_file():
        raise FileNotFoundError(f"The json file {json_path} does not exist.")

    with open(json_path, "r") as json_file:
        try:
            existing_data = json.load(json_file, object_hook=path_decoder)
        except json.JSONDecodeError:
            existing_data = {}

    return existing_data


def write_json(
    json_path: PathType, data: Dict[str, Any], overwrite: bool = False
) -> None:
    """
    Writes the serialized config class to a JSON file.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(exist_ok=True, parents=True)

    if json_path.is_file() and not overwrite:
        raise FileExistsError(f"The JSON file already exists: {json_path}")
    elif json_path.is_file() and overwrite:
        json_path.unlink()

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, default=path_encoder)


def path_encoder(obj):
    """
    Recursively convert Path objects to strings in dicts
    where keys suggest they point to filesystem paths.
    """
    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [path_encoder(item) for item in obj]

    # Handle dicts / OrderedDicts
    if isinstance(obj, (dict, OrderedDict)):
        return {key: path_encoder(value) for key, value in obj.items()}

    # Handle torch modules
    if isinstance(obj, torch.nn.Module):
        return obj.__class__.__name__

    # Handle pathlib.Path
    if isinstance(obj, Path):
        return obj.as_posix()

    # Handle enums
    if isinstance(obj, enum.Enum):
        return obj.name

    # Handle basic types (JSON-compatible)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fallback: convert everything else to string
    return str(obj)


def path_decoder(obj):
    """
    Recursively convert JSON-safe values back into Python objects.
    - String paths -> Path
    - Everything else stays as-is
    """
    if isinstance(obj, dict):
        obj2 = deepcopy(obj)
        for key, value in obj2.items():
            obj[key] = path_decoder(value)
        return obj
    elif isinstance(obj, list):
        return [path_decoder(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(path_decoder(v) for v in obj)
    else:
        return obj
