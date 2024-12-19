from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[Path, str]

DataType = Union[PathLike, pd.DataFrame]
