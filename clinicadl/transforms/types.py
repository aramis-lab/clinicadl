from typing import Callable, Tuple, Union

from pydantic import NonNegativeFloat

from clinicadl.data.structures import DataPoint

from .config import TransformConfig

Transform = Callable[[DataPoint], DataPoint]

TransformOrConfig = Union[Transform, TransformConfig]

Std = Union[
    NonNegativeFloat,
    Tuple[NonNegativeFloat, NonNegativeFloat],
    Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
    Tuple[
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
    ],
]


SpatialRange = Union[
    NonNegativeFloat,
    tuple[float, float],
    Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat],
    Tuple[float, float, float, float, float, float],
]
