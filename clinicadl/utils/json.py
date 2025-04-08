import inspect
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict

from pydantic import BaseModel, ConfigDict, computed_field

from clinicadl.dictionary.words import NAME
from clinicadl.utils.iotools.utils import path_decoder, path_encoder


def read_json(json_path: Path) -> Dict[str, Any]:
    """
    Reads the serialized config class from a JSON file.
    """

    if not json_path.is_file():
        raise FileNotFoundError(f"This json file {json_path} does not exist.")

    with open(json_path, "r") as json_file:
        try:
            existing_data = json.load(json_file, default=path_decoder)
        except json.JSONDecodeError:
            existing_data = {}

    return existing_data


def write_json(json_path: Path, data: Dict[str, Any], overwrite: bool = False) -> None:
    """
    Writes the serialized config class to a JSON file.
    """

    if json_path.is_file() and not overwrite:
        raise FileExistsError(f"The JSON file already exists: {json_path}")
    elif json_path.is_file() and overwrite:
        json_path.unlink()

    with open(json_path, "w") as json_file:
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
    if isinstance(obj, Path):
        return obj.as_posix

    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                obj[key] = path_encoder(value)
            elif is_path_key(key):
                if not value:
                    obj[key] = ""
                elif isinstance(value, Path):
                    obj[key] = value.as_posix()
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
            if isinstance(value, dict):
                obj[key] = path_decoder(value)
            elif is_path_key(key):
                if value in ("", False, None):
                    obj[key] = False
                else:
                    obj[key] = Path(value)
    return obj
