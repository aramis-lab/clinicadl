from typing import Callable, TypeVar

IS_MANDATORY = "_is_mandatory"

F = TypeVar("F", bound=Callable)


def mandatory(func: F) -> F:
    """
    Decorator to define paths that must exist
    when reading a directory.
    """
    setattr(func, IS_MANDATORY, True)
    return func
