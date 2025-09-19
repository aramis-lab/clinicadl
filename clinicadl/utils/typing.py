from pathlib import Path
from typing import Union

import pandas as pd

# TODO : do we keep a typing file or do we remove it and put the types where they are used ?
# If we keep it, we should probably add more types here
PathType = Union[Path, str]
DataType = Union[PathType, pd.DataFrame]
