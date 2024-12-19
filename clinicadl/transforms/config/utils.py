from typing import Sequence, Tuple, Union

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


def is_sorted(seq: Sequence) -> bool:
    """Checks if a sequence is sorted.

    Parameters
    ----------
    seq : Sequence
        the sequence.

    Returns
    -------
    bool
        Whether the sequence is sorted.
    """
    return sorted(list(seq)) == list(seq)
