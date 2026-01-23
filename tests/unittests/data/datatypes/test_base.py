import json
import re

import pytest
from pydantic import ValidationError

from clinicadl.data.datatypes import DataType


def test_datatype(tmp_path):
    data_type = DataType(
        pattern=".*/abc_.*", key="my_datatype", description="A description"
    )
    assert data_type.pattern == re.compile(".*/abc_.*")
    assert data_type.key == "my_datatype"
    assert data_type.name == "DataType"
    assert data_type.description == "A description"
    assert data_type.json_filename == "default_my_datatype.json"
    assert data_type.tsv_filename == "overview_my_datatype.tsv"

    data_type = DataType(pattern="abc", key="abc")
    assert data_type.pattern == re.compile("abc")

    with pytest.raises(ValidationError):
        DataType(pattern="abc", key="abc d")

    data_type = DataType.from_folder_and_suffix(
        "my_folder", suffix="example", description="A description"
    )
    assert data_type.pattern == re.compile("my_folder/sub-.*_ses-.*_example.nii.*")
    assert data_type.key == "example"
    assert data_type.description == "A description"
    assert data_type.json_filename == "default_example.json"
    assert data_type.tsv_filename == "overview_example.tsv"

    patterns = {
        "my_folder/sub-.*_ses-.*_example.nii": True,
        "my_folder/sub-.*_ses-.*_example.nii.gz": True,
        "my_folder/sub-.*_ses-.*_abc.nii": False,
        "abc/sub-.*_ses-.*_example.nii.gz": False,
    }
    for pattern, match in patterns.items():
        assert (data_type.pattern.match(pattern) is not None) == match

    # json
    data_type = DataType(
        pattern=".*/abc_.*", key="my_datatype", description="A description"
    )
    with open(tmp_path / "datatype.json", "w") as f:
        json.dump(data_type.to_dict(), f)
    with open(tmp_path / "datatype.json", "r") as f:
        dict_ = json.load(f)
    data_type = DataType.from_dict(dict_)
    assert data_type.pattern == re.compile(".*/abc_.*")

    # equality
    data_type_1 = DataType(
        pattern=".*/abc_.*", key="my_datatype", description="A description"
    )

    class DataTypeChild(DataType):
        pass

    data_type_2 = DataTypeChild(
        pattern=".*/abc_.*", key="my_data", description="Another description"
    )
    assert not data_type_1 == data_type_2
    data_type_2.key = "my_datatype"
    assert data_type_1 == data_type_2
    assert not data_type_1 == 0
