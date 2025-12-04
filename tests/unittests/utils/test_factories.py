import json
from typing import Any

from clinicadl.utils.factories import (
    factory_from_dict,
    factory_from_json,
    get_args_from,
    get_defaults_from,
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
    def get_obj_from_dict(data: dict[str, Any]) -> Obj:
        """
        A doc.
        """

    @factory_from_dict(
        object_type=ObjConfig, enum=ImplementedObj, context=globals(), config=True
    )
    def get_objconfig_from_dict(data: dict[str, Any]) -> ObjConfig:
        """
        A doc.
        """

    obj = get_obj_from_dict(data={"name": "ObjA", "a": 0})
    assert isinstance(obj, Obj)
    assert obj.a == 0

    obj = get_objconfig_from_dict({"name": "ObjA", "a": 0})
    assert isinstance(obj, ObjConfig)
    assert obj.a == 0


def test_factory_from_json(tmp_path):
    with open(tmp_path / "data.json", "w") as f:
        json.dump({"name": "ObjA", "a": 0}, f)

    @factory_from_json(
        object_type=Obj, enum=ImplementedObj, context=globals(), config=False
    )
    def get_obj_from_json(data: dict[str, Any]) -> Obj:
        """
        A doc.
        """

    @factory_from_json(
        object_type=ObjConfig, enum=ImplementedObj, context=globals(), config=True
    )
    def get_objconfig_from_json(data: dict[str, Any]) -> ObjConfig:
        """
        A doc.
        """

    obj = get_obj_from_json(data=tmp_path / "data.json")
    assert isinstance(obj, Obj)
    assert obj.a == 0

    obj = get_objconfig_from_json(tmp_path / "data.json")
    assert isinstance(obj, ObjConfig)
    assert obj.a == 0
