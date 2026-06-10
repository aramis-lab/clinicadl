import json
from pathlib import Path
from typing import Any, Optional

from clinicadl.utils.factories import (
    factory_from_dict,
    factory_from_json,
    get_args_from,
    get_defaults_from,
    safe_factory_from_json,
)

from .utils import *


def f(a, b="b", c=0, d=None):
    return None


def test_get_defaults_from():
    defaults = get_defaults_from(f)
    assert defaults == {"b": "b", "c": 0, "d": None}


def test_get_args_from():
    args = get_args_from(f)
    assert args == ["a", "b", "c", "d"]


def test_factory_from_dict():
    @factory_from_dict(
        object_type=Obj, enum=ImplementedObj, context=globals(), config=False
    )
    def get_obj_from_dict(data: dict[str, Any], **kwargs) -> Obj:
        """A doc."""

    assert get_obj_from_dict.__doc__ == """A doc."""
    obj = get_obj_from_dict(data={"name_": "ObjA", "a": 0, "b": 1})
    assert obj.a == 0
    assert obj.b == 1
    assert isinstance(obj, ObjA)
    obj = get_obj_from_dict(data={"name_": "ObjA", "a": 0, "b": 1}, b=2)
    assert obj.b == 2

    @factory_from_dict(
        object_type=ObjConfig, enum=ImplementedConfig, context=globals(), config=True
    )
    def get_config_from_dict(data: dict[str, Any], **kwargs) -> ObjConfig:
        pass

    obj = get_config_from_dict(data={"name_": "ObjB", "a": 0, "b": 1})
    assert obj.a == 0
    assert obj.b == 1
    assert isinstance(obj, ObjBConfig)

    @factory_from_dict(
        object_type=NamedConfig,
        enum=ImplementedNamedConfig,
        context=globals(),
        config=False,
    )
    def get_config_from_dict(data: dict[str, Any], **kwargs) -> NamedConfig:
        pass

    obj = get_config_from_dict(data={"name_": "ConfigC", "a": 0, "b": 1})
    assert obj.a == 0
    assert obj.b == 1
    assert isinstance(obj, ConfigC)


def test_factory_from_json(tmp_path):
    with open(tmp_path / "data.json", "w") as f:
        json.dump({"name_": "ObjA", "a": 0, "b": 1}, f)

    @factory_from_json(
        object_type=Obj, enum=ImplementedObj, context=globals(), config=False
    )
    def get_obj_from_json(data: Path, **kwargs) -> Obj:
        """A doc."""

    assert get_obj_from_json.__doc__ == """A doc."""
    obj = get_obj_from_json(data=tmp_path / "data.json")
    assert obj.a == 0
    assert obj.b == 1
    assert isinstance(obj, ObjA)
    obj = get_obj_from_json(data=tmp_path / "data.json", b=2)
    assert obj.b == 2

    @factory_from_json(
        object_type=ObjConfig, enum=ImplementedConfig, context=globals(), config=True
    )
    def get_config_from_json(data: Path, **kwargs) -> ObjConfig:
        pass

    with open(tmp_path / "data.json", "w") as f:
        json.dump({"name_": "ObjB", "a": 0, "b": 1}, f)

    obj = get_config_from_json(data=tmp_path / "data.json")
    assert obj.a == 0
    assert obj.b == 1
    assert isinstance(obj, ObjBConfig)

    @factory_from_json(
        object_type=NamedConfig,
        enum=ImplementedNamedConfig,
        context=globals(),
        config=False,
    )
    def get_config_from_json(data: Path, **kwargs) -> NamedConfig:
        pass

    with open(tmp_path / "data.json", "w") as f:
        json.dump({"name_": "ConfigC", "a": 0, "b": 1}, f)

    obj = get_config_from_json(data=tmp_path / "data.json")
    assert obj.a == 0
    assert obj.b == 1
    assert isinstance(obj, ConfigC)


def test_safe_factory_from_json(tmp_path):
    with open(tmp_path / "data.json", "w") as f:
        json.dump({"name_": "ObjA", "a": 0, "b": "x"}, f)

    @factory_from_json(
        object_type=Obj, enum=ImplementedObj, context=globals(), config=False
    )
    def get_obj_from_json(data: Path, **kwargs) -> Obj:
        pass

    @safe_factory_from_json(factory=get_obj_from_json)
    def get_obj_from_json_safe(
        data: Path, default: Optional[Obj] = None
    ) -> tuple[Optional[Obj], list[str]]:
        """A doc."""

    assert get_obj_from_json_safe.__doc__ == """A doc."""
    assert get_obj_from_json_safe(data=tmp_path / "data.json") == (None, [])

    out, fields = get_obj_from_json_safe(
        data=tmp_path / "data.json", default=ObjA(a=1, b=1)
    )
    assert fields == ["b"]
    assert out.a == 0
    assert out.b == 1

    ###
    with open(tmp_path / "data.json", "w") as f:
        json.dump({"name_": "ObjA", "a": "x", "b": "x"}, f)

    out, fields = get_obj_from_json_safe(
        data=tmp_path / "data.json", default=ObjA(a=1, b=1)
    )
    assert fields == ["a", "b"]
    assert out.a == 1
    assert out.b == 1

    ###
    with open(tmp_path / "data.json", "w") as f:
        json.dump({"name_": "ObjX", "a": 1, "b": 1}, f)

    assert get_obj_from_json_safe(data=tmp_path / "data.json") == (None, [])

    ###
    with open(tmp_path / "data.json", "w") as f:
        json.dump({"a": 1, "b": 1}, f)

    assert get_obj_from_json_safe(data=tmp_path / "data.json") == (None, [])

    ###
    @safe_factory_from_json(factory=SimpleConfig.from_json)
    def get_obj_from_json_safe(
        data: Path, default: Optional[Obj] = None
    ) -> tuple[Optional[Obj], list[str]]:
        pass

    with open(tmp_path / "data.json", "w") as f:
        json.dump({"a": "x", "b": "x"}, f)

    out, fields = get_obj_from_json_safe(
        data=tmp_path / "data.json", default=SimpleConfig(a=1, b=1)
    )
    assert fields == ["a", "b"]
    assert out.a == 1
    assert out.b == 1
