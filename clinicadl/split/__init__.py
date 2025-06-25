"""To split data into training, validation and test sets."""

from .make_splits import make_kfold, make_split
from .split import Split
from .splitter import KFold, SingleSplit
