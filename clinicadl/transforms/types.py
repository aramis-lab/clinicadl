from typing import Callable, TypeVar, Union

from clinicadl.data.structures import DataPoint

from .config import TransformConfig

DataPointT = TypeVar("DataPointT", bound=DataPoint)


Transform = Callable[[DataPointT], DataPointT]

TransformOrConfig = Union[Transform, TransformConfig]
