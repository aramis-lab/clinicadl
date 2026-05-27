"""To build a :py:class:`PyTorch DataLoader <torch.utils.data.DataLoader>`
adapted to ``ClinicaDL``."""

from .batch import Batch, BatchType
from .collate import *
from .loader import DataLoader
