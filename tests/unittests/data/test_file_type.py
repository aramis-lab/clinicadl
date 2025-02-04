import pytest

from clinicadl.data.datatype.preprocessing.file_type import FileType
from clinicadl.data.datatype.utils import PreprocessingMethod


def test_good_filetype():
    config = FileType(
        pattern="test",
        description="file type configurztion for unittests",
        needed_pipeline="t1-linear",  # type: ignore
    )

    assert config.pattern == "test"
    assert config.description == "file type configurztion for unittests"
    assert isinstance(config.needed_pipeline, PreprocessingMethod)
    assert config.needed_pipeline == PreprocessingMethod.T1_LINEAR


def test_bad_filetype():
    with pytest.raises(ValueError):
        FileType(pattern="/test", description="test")

    with pytest.raises(ValueError):
        FileType(pattern="test", needed_pipeline="t1-linear")  # type: ignore

    with pytest.raises(ValueError):
        FileType(
            pattern="test",
            description="test",
            needed_pipeline="invalid_pipeline",  # type: ignore
        )
