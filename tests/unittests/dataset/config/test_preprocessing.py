from pathlib import Path

import pytest
from pydantic import ValidationError

from clinicadl.dataset.config.preprocessing import (
    PreprocessingCustom,
    PreprocessingDTI,
    PreprocessingFlair,
    PreprocessingPET,
    PreprocessingT1,
    PreprocessingT2,
)
