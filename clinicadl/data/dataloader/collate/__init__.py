"""
To define how the ``DataLoader`` will collate the collections of samples.
"""

from .base import CollateFn
from .merge_batches import MergeBatches
from .to_batch import ToBatch
from .to_batches import ToBatches
