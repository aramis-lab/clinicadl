"""To build a :py:class:`PyTorch Dataset <torch.utils.data.Dataset>`
with your neuroimaging data."""

from .caps_dataset import CapsDataset
from .concat import ConcatDataset
from .paired import PairedDataset
from .unpaired import UnpairedDataset
