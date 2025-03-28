from pathlib import Path

import pandas as pd
import pytest

from clinicadl.maps import Maps
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
)

maps_path = Path()

def test_good_maps():
    maps = Maps(maps_path)
    