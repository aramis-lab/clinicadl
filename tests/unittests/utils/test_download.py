import os
from pathlib import Path
from unittest.mock import patch

from clinicadl.utils.download import get_clinicadl_cache_dir


@patch(
    "clinicadl.utils.download.user_cache_dir",
    side_effect=lambda x: os.path.join("cache", x),
)
def test_get_clinicadl_cache_dir(cache_mock):
    assert get_clinicadl_cache_dir() == Path("cache") / "clinicadl"
