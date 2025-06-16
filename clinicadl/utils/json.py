import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import torch


def read_json(json_path: Path) -> Dict[str, Any]:
    """
    Reads the serialized config class from a JSON file.
    """

    if not json_path.is_file():
        raise FileNotFoundError(f"The json file {json_path} does not exist.")

    with open(json_path, "r") as json_file:
        try:
            existing_data = json.load(json_file, object_hook=path_decoder)
        except json.JSONDecodeError:
            existing_data = {}

    return existing_data


def write_json(json_path: Path, data: Dict[str, Any], overwrite: bool = False) -> None:
    """
    Writes the serialized config class to a JSON file.
    """
    json_path.parent.mkdir(exist_ok=True, parents=True)

    if json_path.is_file() and not overwrite:
        raise FileExistsError(f"The JSON file already exists: {json_path}")
    elif json_path.is_file() and overwrite:
        json_path.unlink()

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, default=path_encoder)


def update_json(json_path: Path, new_data: Dict[str, Any]) -> None:
    """
    Updates the JSON file with the serialized config class.
    """

    # Lire le contenu existent du fichier
    existing_data = read_json(json_path)

    # Fusionner les nouvelles données
    existing_data.update(new_data)

    # Écrire les données mises à jour dans le fichier
    write_json(json_path, existing_data, overwrite=True)


def is_path_key(key: str) -> bool:
    """Check if a key is likely to refer to a path."""
    path_keywords = ("tsv", "dir", "directory", "path", "json", "location")
    return any(key.lower().endswith(suffix) for suffix in path_keywords)


def path_encoder(obj):
    """
    Recursively convert Path objects to strings in dicts
    where keys suggest they point to filesystem paths.
    """
    if isinstance(obj, list):
        return [path_encoder(item) for item in obj]
    if isinstance(obj, torch.nn.modules.Module):
        return obj.__class__.__name__
    if isinstance(obj, Path):
        return obj.as_posix()

    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = path_encoder(value)
        return obj

    return obj


def path_decoder(obj):
    """
    Recursively convert string/empty/False values to Path objects in dicts
    where keys suggest they refer to filesystem paths.
    """
    if isinstance(obj, dict):
        obj2 = deepcopy(obj)
        for key, value in obj2.items():
            if is_path_key(key):
                if value in ("", False, None):
                    obj2[key] = False
                else:
                    obj2[key] = Path(value)
            else:
                obj2[key] = path_decoder(value)
        return obj2
    else:
        return obj
