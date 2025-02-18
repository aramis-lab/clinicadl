from collections import OrderedDict
from typing import Union

from clinicadl.utils.config import ClinicaDLConfig


def test_to_dict():
    class SubConfigTest(ClinicaDLConfig):
        a: str = "a"
        name: str = "SubTest"

    class ConfigTest(ClinicaDLConfig):
        list_test: list[Union[str, SubConfigTest]] = ["list", SubConfigTest()]
        tuple_test: tuple[SubConfigTest, SubConfigTest] = (
            SubConfigTest(),
            SubConfigTest(),
        )
        name: str = "Config"

    config = ConfigTest()
    ordered_subtest = OrderedDict(name="SubTest", a="a")
    assert config.to_dict() == OrderedDict(
        **{
            "name": "Config",
            "list_test": ["list", ordered_subtest],
            "tuple_test": (ordered_subtest, ordered_subtest),
        }
    )
