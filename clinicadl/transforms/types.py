from typing import Callable, Tuple, Union

from pydantic import NonNegativeFloat

from clinicadl.data.structures import DataPoint

Transform = Callable[[DataPoint], DataPoint]

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
