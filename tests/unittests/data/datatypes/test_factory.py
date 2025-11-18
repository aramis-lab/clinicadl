import pytest

from clinicadl.data.datatypes import *
from clinicadl.data.datatypes.factory import get_datatype_from_dict


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
def test_get_datatype_from_dict(args, datatype):
    c = datatype(**args)
    dict_ = c.to_dict()
    print(dict_)
    c = get_datatype_from_dict(dict_)
    assert isinstance(c, datatype)

    if datatype is DataType:
        c = datatype.from_folder_and_suffix(folder="abc", suffix="bcd")
        new_c = get_datatype_from_dict(c.to_dict())
        assert c.key == "bcd"
