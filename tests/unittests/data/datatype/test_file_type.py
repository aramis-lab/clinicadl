import pytest

from clinicadl.data.datatypes.file_type import FileType


def test_good_filetype():
    config = FileType(
        pattern="test",
        description="file type configurztion for unittests",
        needed_pipeline="t1-linear",
    )

    assert config.pattern == "test"
    assert config.description == "file type configurztion for unittests"
    assert config.needed_pipeline == "t1-linear"


def test_bad_filetype():
    with pytest.raises(ValueError):
        FileType(pattern="/test", description="test")

    with pytest.raises(ValueError):
        FileType(pattern="test", needed_pipeline="t1-linear")
