from enum import Enum


class Optimum(str, Enum):
    """Optimization criterion for a metric."""

    MIN = "min"
    MAX = "max"
