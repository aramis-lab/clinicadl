from pathlib import Path

import pytest
from pydantic import ValidationError

from clinicadl.dataset.config.file_type import FileType

BAD_INPUTS = [
    ("", "", ""),
    ("/my_path/to/pattern", "file type for t1linear preprocessing", "t1-linear"),
    ("*", "", "t2-linear"),
    ("*tt*", "file type for my own pipeline", "my-pipeline"),
]


@pytest.mark.parametrize(
    "pattern,description,needed",
    BAD_INPUTS,
)
def test_bad_filetype(pattern, description, needed):
    print(pattern)
    print(description)
    print(needed)
    # FileType(pattern=pattern, description=description, needed_pipeline=needed)
    with pytest.raises(ValidationError):
        FileType(pattern=pattern, description=description, needed_pipeline=needed)


GOOD_INPUTS = [
    ("my_path/to/pattern", "file type for t1linear preprocessing", "t1-linear"),
    ("*T2.nii*", "file type for t2-linear ", "t2-linear"),
    ("*tt*", "file type for my own pipeline", "custom"),
]


@pytest.mark.parametrize(
    "pattern,description,needed",
    GOOD_INPUTS,
)
def test_good_filetype(pattern, description, needed):
    test = FileType(pattern=pattern, description=description, needed_pipeline=needed)
    assert test.pattern == pattern
    assert test.description == description
    assert test.needed_pipeline == needed
