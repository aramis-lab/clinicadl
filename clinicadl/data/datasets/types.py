from typing import Union

from .caps_dataset import CapsDataset
from .concat import ConcatDataset
from .paired import PairedDataset
from .unpaired import UnpairedDataset

SimpleDataset = Union[CapsDataset, ConcatDataset]
TupleDataset = Union[PairedDataset, UnpairedDataset]
Dataset = Union[SimpleDataset, TupleDataset]
