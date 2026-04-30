"""To build a :py:class:`PyTorch Dataset <torch.utils.data.Dataset>`
with your neuroimaging data."""

from .base import Dataset
from .bids import BidsDataset
from .tensor import TensorDataset
# from .caps import CapsDataset
# from .concat import ConcatDataset
# from .multi_samples import MultiSamplesDataset
# from .paired import PairedDataset
# from .sampler import SamplerDataset
# from .unpaired import UnpairedDataset
