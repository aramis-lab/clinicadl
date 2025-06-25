"""To build a :py:class:`PyTorch Dataset <torch.utils.data.Dataset>`
for neuroimaging data stored in a :term:`CAPS` structure."""

from .caps_dataset import CapsDataset
from .concat import ConcatDataset
from .paired import PairedDataset
from .unpaired import UnpairedDataset
