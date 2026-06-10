from pathlib import Path
from typing import Union

import pandas as pd

PathType = Union[Path, str]
DataFrameType = Union[PathType, pd.DataFrame]
