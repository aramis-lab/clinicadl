import pytest

from clinicadl.dataset.config import DataConfig, FileType
from clinicadl.utils.enum import Preprocessing


def test_good_filetype():
    config = FileType(
        pattern="test",
        description="file type configurztion for unittests",
        needed_pipeline="t1-linear",
    )

    assert config.pattern == "test"
    assert config.description == "file type configurztion for unittests"
    assert isinstance(config.needed_pipeline, Preprocessing)
    assert config.needed_pipeline == Preprocessing.T1_LINEAR


def test_bad_filetype():
    with pytest.raises(ValueError):
        config = FileType(pattern="/test", description="test")

    with pytest.raises(ValueError):
        config = FileType(pattern="test", needed_pipeline="t1-linear")

    with pytest.raises(ValueError):
        config = FileType(
            pattern="test", description="test", needed_pipeline="invalid_pipeline"
        )
