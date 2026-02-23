import pytest

from clinicadl.data.datatypes import *
from clinicadl.data.datatypes.factory import get_datatype_from_dict
from clinicadl.utils.json import read_json


@pytest.mark.parametrize(
    "args, datatype",
    [
        ({}, T1Linear),
        ({}, FlairLinear),
        ({"tracer": "18FFDG", "suvr_reference_region": "pons2"}, PETLinear),
        ({"measure": "AD", "space": "native"}, DWIDTI),
        ({"pattern": ".*.nii", "key": "anything"}, DataType),
    ],
)
def test_get_datatype_from_dict(args, datatype, tmp_path):
    c = datatype(**args)
    c.to_json(tmp_path / "config.json")
    dict_ = read_json(tmp_path / "config.json")
    c = get_datatype_from_dict(dict_)
    assert isinstance(c, datatype)

    if datatype is DataType:
        c = datatype.from_folder_and_suffix(folder="abc", suffix="bcd")
        new_c = get_datatype_from_dict(c.to_dict())
        assert c.key == "bcd"
