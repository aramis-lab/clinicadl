from __future__ import annotations

from enum import Enum

from pydantic import Field

from clinicadl.utils.config import ClinicaDLConfig, ConfigWithName, ObjectConfig
from clinicadl.utils.objects import HasConfig, JsonReaderWriter, Serializable


class Obj(JsonReaderWriter, Serializable):
    a: int
    b: int


class ObjAConfig(ObjectConfig[Obj]):
    a: int = Field(json_schema_extra={"reader": lambda x: int(x)})
    b: int = Field(json_schema_extra={"reader": lambda x: int(x)})

    @classmethod
    def _get_class(cls):
        return ObjA


class ObjA(Obj, HasConfig["ObjAConfig"]):
    _config_type = ObjAConfig

    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b
        self.config = self._config_type(a=a, b=b)


class ImplementedObj(str, Enum):
    A = "ObjA"


###


class ObjConfig(ObjectConfig):
    pass


class ObjB:
    pass


class ObjBConfig(ObjectConfig):
    a: int
    b: int

    @classmethod
    def _get_class(cls):
        return ObjB


class ImplementedConfig(str, Enum):
    B = "ObjB"


###


class NamedConfig(ConfigWithName):
    pass


class ConfigC(NamedConfig):
    a: int
    b: int


class ImplementedNamedConfig(str, Enum):
    C = "ConfigC"


###


class SimpleConfig(ClinicaDLConfig):
    a: int = Field(json_schema_extra={"reader": lambda x: int(x)})
    b: int = Field(json_schema_extra={"reader": lambda x: int(x)})
