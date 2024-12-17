from typing import Tuple, Union

from pydantic import NonNegativeInt

Bounds = Union[
    NonNegativeInt,
    Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt],
    Tuple[
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
    ],
]
