"""
To define how the ``DataLoader`` will collate the collections of samples.
"""

from .base import CollateFn
from .merge_batches import MergeBatchesCollate
from .to_batch import ToBatchCollate
from .to_batches import ToBatchesCollate
