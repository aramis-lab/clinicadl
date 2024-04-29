from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel


class EnumTest(str, Enum):
    OPTION1 = "option1"
    OPTION2 = "option2"


class ConfigTest(BaseModel):
    parameter_str: str = "a string"
    parameter_int: int = 0
    parameter_float: float = 0.0
    parameter_path: Path = Path("a/path")
    parameter_bool: bool = True
    parameter_list: List[int] = [0, 1]
    parameter_tuple: Tuple[str, ...] = ("elem1", "elem2")
    parameter_empty_tuple: Tuple[str, ...] = ()
    parameter_enum: EnumTest = EnumTest.OPTION1
    parameter_enum_optional: Optional[EnumTest] = None


def test_get_default_from_config_class():
    from clinicadl.utils.config_utils import get_default_from_config_class

    test_config = ConfigTest()
    assert get_default_from_config_class("parameter_str", test_config) == "a string"
    assert get_default_from_config_class("parameter_int", test_config) == 0
    assert get_default_from_config_class("parameter_float", test_config) == 0.0
    assert get_default_from_config_class("parameter_path", test_config) == Path(
        "a/path"
    )
    assert get_default_from_config_class("parameter_bool", test_config)
    assert get_default_from_config_class("parameter_list", test_config) == [0, 1]
    assert get_default_from_config_class("parameter_tuple", test_config) == (
        "elem1",
        "elem2",
    )
    assert get_default_from_config_class("parameter_empty_tuple", test_config) == ()
    assert get_default_from_config_class("parameter_enum", test_config) == "option1"
    assert get_default_from_config_class("parameter_enum_optional", test_config) is None


def test_get_type_from_config_class():
    from clinicadl.utils.config_utils import get_type_from_config_class

    test_config = ConfigTest()
    assert get_type_from_config_class("parameter_str", test_config) == str
    assert get_type_from_config_class("parameter_int", test_config) == int
    assert get_type_from_config_class("parameter_float", test_config) == float
    assert get_type_from_config_class("parameter_path", test_config) == Path
    assert get_type_from_config_class("parameter_bool", test_config) == bool
    assert get_type_from_config_class("parameter_list", test_config) == int
    assert get_type_from_config_class("parameter_tuple", test_config) == str
    assert get_type_from_config_class("parameter_empty_tuple", test_config) == str
    assert get_type_from_config_class("parameter_enum", test_config) == [
        "option1",
        "option2",
    ]
    assert get_type_from_config_class("parameter_enum_optional", test_config) == [
        "option1",
        "option2",
    ]
