import json
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Sequence, Union

import pytest
from pydantic import Field, field_validator
from typing_extensions import Self

from clinicadl.utils.config import (
    ClinicaDLConfig,
    ObjectConfig,
    ObjectOrConfig,
    SequenceOfObjects,
)
from clinicadl.utils.exceptions import (
    CannotReadJsonFieldError,
    MissingFieldsJsonError,
    NotInterpretableJsonError,
    WrongFieldsJsonError,
)
from clinicadl.utils.json import read_json


class ConfigTest(ClinicaDLConfig):
    a: int = 0


class ConfigTestBis(ConfigTest):
    pass


class ObjectTest:
    def __init__(self, a: int):
        self.a: int = a


class ObjectTestConfig(ObjectConfig[ObjectTest], ConfigTest):
    @classmethod
    def _get_class(cls) -> Any:
        return ObjectTest


class ClinicaDLObjectTest(ObjectTest):
    def to_dict(self):
        return self.__dict__ | {"bonus": "abc"}

    @classmethod
    def from_dict(cls, dict_) -> Self:
        d = deepcopy(dict_)
        del d["bonus"]
        return cls(**d)


class CollectionTest:
    def __init__(self, object: ObjectTest, list_objects: Sequence[ObjectTest]):
        self.object = object
        self.list_objects = list_objects


class MainConfigTest(ClinicaDLConfig):
    simple: str
    sub_test: ConfigTest
    object_or_config: Union[ObjectTest, ObjectTestConfig]
    object_or_config_bis: ObjectOrConfig[ObjectTest, ObjectTestConfig] = Field(
        reader=ObjectOrConfig.build_reader(ObjectTestConfig.from_dict)
    )
    object: ClinicaDLObjectTest = Field(reader=ClinicaDLObjectTest.from_dict)
    list_configs: SequenceOfObjects[ObjectTest, ObjectTestConfig] = Field(
        reader=SequenceOfObjects.build_reader(ObjectTestConfig.from_dict)
    )

    @field_validator("object_or_config_bis", mode="before")
    @classmethod
    def _handle_any_value(cls, v: Any) -> ObjectOrConfig:
        return ObjectOrConfig.from_value(v)

    @field_validator("list_configs", mode="before")
    @classmethod
    def _handle_sequence(cls, v: Any) -> SequenceOfObjects:
        return SequenceOfObjects.from_sequence(v, field_name="list_configs")


class CollectionTestConfig(ObjectConfig[CollectionTest]):
    object: ObjectOrConfig[ObjectTest, ObjectTestConfig] = Field(
        reader=ObjectOrConfig.build_reader(ObjectTestConfig.from_dict)
    )
    list_objects: SequenceOfObjects[ObjectTest, ObjectTestConfig] = Field(
        reader=SequenceOfObjects.build_reader(ObjectTestConfig.from_dict)
    )

    @field_validator("object", mode="before")
    @classmethod
    def _handle_any_value(cls, v: Any) -> ObjectOrConfig:
        return ObjectOrConfig.from_value(v)

    @field_validator("list_objects", mode="before")
    @classmethod
    def _handle_sequence(cls, v: Any) -> SequenceOfObjects:
        return SequenceOfObjects.from_sequence(v, field_name="list_configs")

    @classmethod
    def _get_class(cls) -> type[CollectionTest]:
        return CollectionTest


def test_clinicadl_config(tmp_path):
    params = {
        "simple": "str",
        "sub_test": ConfigTest(a=1),
        "object_or_config": ObjectTestConfig(a=-1),
        "object_or_config_bis": ObjectTestConfig(a=-1),
        "object": ClinicaDLObjectTest(a=-1),
        "list_configs": [ObjectTestConfig(a=0), ObjectTestConfig(a=1)],
    }
    t = MainConfigTest(**params)

    # to json
    t.to_json(tmp_path / "config.json", overwrite=True)
    dict_ = read_json(tmp_path / "config.json")
    assert dict_["simple"] == "str"
    assert dict_["sub_test"] == {"a": 1}
    assert OrderedDict(dict_["object_or_config"]) == OrderedDict(
        {"name": "ObjectTest", "a": -1}
    )
    assert OrderedDict(dict_["object_or_config_bis"]) == OrderedDict(
        {"name": "ObjectTest", "a": -1}
    )
    assert dict_["object"] == {"a": -1, "bonus": "abc"}
    assert [OrderedDict(d) for d in dict_["list_configs"]] == [
        OrderedDict({"name": "ObjectTest", "a": 0}),
        OrderedDict({"name": "ObjectTest", "a": 1}),
    ]

    # from json
    new_t = MainConfigTest.from_json(
        tmp_path / "config.json",
    )
    assert new_t.simple == "str"
    assert new_t.sub_test.a == 1
    assert new_t.object_or_config.a == -1
    assert new_t.object_or_config_bis.value.a == -1
    assert new_t.object.a == -1
    assert new_t.list_configs.values[0].value.a == 0
    assert new_t.list_configs.values[1].value.a == 1

    # cannot read errors
    t = MainConfigTest(
        simple="str",
        sub_test=ConfigTest(a=1),
        object_or_config=ObjectTest(a=-1),
        object_or_config_bis=ObjectTest(a=-1),
        object=ClinicaDLObjectTest(a=-1),
        list_configs=[ObjectTestConfig(a=0), ObjectTest(a=1)],
    )
    t.to_json(tmp_path / "config.json", overwrite=True)

    new_t = MainConfigTest.from_json(
        tmp_path / "config.json",
        sub_test=ConfigTest(a=2),
        object_or_config=ObjectTest(a=-1),
        object_or_config_bis=ObjectTest(a=-1),
        list_configs=[ObjectTestConfig(a=0), ObjectTest(a=1)],
    )
    assert new_t.simple == "str"
    assert new_t.sub_test.a == 2
    assert new_t.object_or_config.a == -1
    assert new_t.object_or_config_bis.value.a == -1
    assert new_t.object.a == -1
    assert new_t.list_configs.values[0].value.a == 0
    assert new_t.list_configs.values[1].value.a == 1

    with pytest.raises(
        CannotReadJsonFieldError,
        match="MainConfigTest cannot read 'object_or_config' in .*\nPlease pass this field via kwargs.",
    ):
        t = MainConfigTest.from_json(
            tmp_path / "config.json",
            object_or_config_bis=ObjectTest(a=-1),
            list_configs=[ObjectTestConfig(a=0), ObjectTest(a=1)],
        )

    with pytest.raises(
        CannotReadJsonFieldError,
        match="MainConfigTest cannot read 'object_or_config_bis' in .*\nPlease pass this field via kwargs.",
    ):
        t = MainConfigTest.from_json(
            tmp_path / "config.json",
            object_or_config=ObjectTest(a=-1),
            list_configs=[ObjectTestConfig(a=0), ObjectTest(a=1)],
        )

    with pytest.raises(
        CannotReadJsonFieldError,
        match="MainConfigTest cannot read 'list_configs' in .*\nPlease pass this field via kwargs.",
    ):
        MainConfigTest.from_json(
            tmp_path / "config.json",
            object_or_config=ObjectTest(a=-1),
            object_or_config_bis=ObjectTest(a=-1),
        )

    with open(tmp_path / "config.json", "r") as f:
        dict_ = json.load(f)
    del dict_["object"]["a"]
    with open(tmp_path / "config.json", "w") as f:
        json.dump(dict_, f)

    with pytest.raises(
        CannotReadJsonFieldError,
        match="MainConfigTest cannot read 'object' in .*\nPlease pass this field via kwargs.",
    ):
        MainConfigTest.from_json(
            tmp_path / "config.json",
            object_or_config=ObjectTest(a=-1),
            object_or_config_bis=ObjectTest(a=-1),
            list_configs=[ObjectTestConfig(a=0), ObjectTest(a=1)],
        )


def test_get_object():
    t = CollectionTestConfig(
        object=ObjectTestConfig(a=-1),
        list_objects=[ObjectTestConfig(a=0), ObjectTest(a=1)],
    ).get_object()
    assert t.object.a == -1
    assert isinstance(t.object, ObjectTest)
    assert t.list_objects[0].a == 0
    assert t.list_objects[1].a == 1
    assert isinstance(t.list_objects[0], ObjectTest)
    assert isinstance(t.list_objects[1], ObjectTest)


def test_json_reading_error(tmp_path):
    with open(tmp_path / "config.json", "w") as f:
        json.dump(None, f)

    with pytest.raises(NotInterpretableJsonError, match="ConfigTest cannot read .*"):
        ConfigTest.from_json(tmp_path / "config.json")

    with open(tmp_path / "config.json", "w") as f:
        json.dump({"b": 0}, f)

    with pytest.raises(
        MissingFieldsJsonError, match=r"Fields \['a'\] are missing in .*"
    ):
        ConfigTest.from_json(tmp_path / "config.json")

    with open(tmp_path / "config.json", "w") as f:
        json.dump({"a": 10, "b": 0}, f)

    with pytest.raises(
        WrongFieldsJsonError,
        match=r"Fields \['b'\] in .* are not expected by ConfigTest.",
    ):
        ConfigTest.from_json(tmp_path / "config.json")


def test_to_raw_dict():
    t = CollectionTestConfig(
        object=ObjectTestConfig(a=-1),
        list_objects=[ObjectTestConfig(a=0), ObjectTest(a=1)],
    )
    d = t.to_raw_dict()
    assert isinstance(d["object"], ObjectTestConfig)
    assert isinstance(d["list_objects"][0], ObjectTestConfig)
    assert isinstance(d["list_objects"][1], ObjectTest)
    d = t.to_raw_dict(exclude=["object"])
    assert "object" not in d
