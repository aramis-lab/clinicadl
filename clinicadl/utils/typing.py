from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathType = Union[Path, str]

DataType = Union[PathType, pd.DataFrame]
