from enum import Enum
from pathlib import Path
from typing import Any

from typing_extensions import Self

from clinicadl.utils.json import read_json

__all__ = ["Obj", "ObjA", "ObjConfig", "ObjAConfig", "ImplementedObj"]


class Obj:
    def __init__(self, a):
        self.a = a

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        return cls(a=config_dict["a"])

    @classmethod
    def from_json(cls, json_path: Path) -> Self:
        return cls(a=read_json(json_path)["a"])


class ObjA(Obj):
    pass


class ObjConfig(Obj):
    pass


class ObjAConfig(ObjConfig):
    pass


class ImplementedObj(str, Enum):
    A = "ObjA"
