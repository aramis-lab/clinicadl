from typing import Callable, Union

from clinicadl.data.structures import DataPoint

from .config import TransformConfig

Transform = Callable[[DataPoint], DataPoint]

TransformOrConfig = Union[Transform, TransformConfig]
