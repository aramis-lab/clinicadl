from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import Field, field_validator

from clinicadl.utils.config import ObjectConfig, ObjectOrConfig, SequenceOfObjects
from clinicadl.utils.exceptions import CannotReadJsonFieldError
from clinicadl.utils.objects import (
    HasConfig,
    JsonReaderWriter,
    Serializable,
    equal_if_config_equal,
    to_json_safe,
)


class CustomReaderWriter(JsonReaderWriter):
    pass


class CustomSerializable(Serializable):
    pass


class ObjectTest:
    def __init__(self, a: int):
        self.a: int = a


class ObjectTestConfig(ObjectConfig[ObjectTest]):
    a: int

    @classmethod
    def _get_class(cls) -> Any:
        return ObjectTest


class ReaderWriterConfig(ObjectConfig["ReaderWriter"]):
    a: int
    obj: ObjectOrConfig[ObjectTest, ObjectTestConfig] = Field(
        reader=ObjectOrConfig.build_reader(ObjectTestConfig.from_dict)
    )
    seq: SequenceOfObjects[ObjectTest, ObjectTestConfig] = Field(
        reader=SequenceOfObjects.build_reader(ObjectTestConfig.from_dict)
    )

    @field_validator("obj", mode="before")
    @classmethod
    def _handle_any_value(cls, v: Any) -> ObjectOrConfig:
        return ObjectOrConfig.from_value(v)

    @field_validator("seq", mode="before")
    @classmethod
    def _handle_sequence(cls, v: Any) -> SequenceOfObjects:
        return SequenceOfObjects.from_sequence(v, field_name="list_configs")

    @classmethod
    def _get_class(cls) -> ReaderWriter:
        return ReaderWriter


@equal_if_config_equal
class ReaderWriter(HasConfig[ReaderWriterConfig]):
    _config_type = ReaderWriterConfig

    def __init__(self, a, obj, seq):
        self.config = ReaderWriterConfig(a=a, obj=obj, seq=seq)


def test_json(tmp_path):
    json_path = tmp_path / "serialized.json"

    obj = CustomReaderWriter()

    with pytest.raises(NotImplementedError):
        obj.to_json(json_path)

    to_json_safe(obj, json_path)
    with open(json_path, "r") as f:
        d = json.load(f)
    assert "<tests.unittests.utils.test_objects.CustomReaderWriter object at" in d

    obj = ReaderWriter(a=0, obj=ObjectTestConfig(a=1), seq=[ObjectTestConfig(a=2)])
    obj.to_json(json_path, overwrite=True)
    with open(json_path, "r") as f:
        d = json.load(f)
    assert d == {
        "name": "ReaderWriter",
        "a": 0,
        "obj": {"a": 1, "name": "ObjectTest"},
        "seq": [{"a": 2, "name": "ObjectTest"}],
    }

    with pytest.raises(NotImplementedError):
        CustomReaderWriter.from_json(json_path)

    obj = ReaderWriter.from_json(json_path)
    assert isinstance(obj, ReaderWriter)
    assert obj.config.a == 0

    obj = ReaderWriter(a=0, obj=ObjectTest(a=1), seq=[ObjectTestConfig(a=2)])
    obj.to_json(json_path, overwrite=True)
    with pytest.raises(CannotReadJsonFieldError):
        ReaderWriter.from_json(json_path)
    obj = ReaderWriter.from_json(json_path, obj=ObjectTest(a=1))
    assert obj.config.obj.value.a == 1


def test_dict():
    obj = CustomSerializable()

    with pytest.raises(NotImplementedError):
        obj.to_dict()

    raw = ObjectTest(a=3)
    obj = ReaderWriter(a=0, obj=raw, seq=[ObjectTestConfig(a=2), raw])
    d = obj.to_dict()
    assert set(d.keys()) == {"name", "a", "obj", "seq"}
    assert d["name"] == "ReaderWriter"
    assert d["a"] == 0
    assert d["obj"] is raw
    assert d["seq"][0] == {"a": 2, "name": "ObjectTest"}
    assert d["seq"][1] is raw

    with pytest.raises(NotImplementedError):
        CustomSerializable.from_dict(d)

    obj = ReaderWriter.from_dict(d)
    assert isinstance(obj, ReaderWriter)
    assert obj.config.a == 0
    assert isinstance(obj.config.obj.value, ObjectTest)
    assert isinstance(obj.config.seq.values[0].value, ObjectTestConfig)
    assert isinstance(obj.config.seq.values[1].value, ObjectTest)


def test_eq():
    r1 = ReaderWriter(a=0, obj=ObjectTestConfig(a=1), seq=[ObjectTestConfig(a=2)])
    r2 = ReaderWriter(a=0, obj=ObjectTestConfig(a=0), seq=[ObjectTestConfig(a=2)])
    assert r1 != r2
    r2.config.obj = ObjectTestConfig(a=1)
    assert r1 == r2

    class ReaderWriterChild(ReaderWriter):
        pass

    r2 = ReaderWriterChild(a=0, obj=ObjectTestConfig(a=1), seq=[ObjectTestConfig(a=2)])
    assert r1 != r2
