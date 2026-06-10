import inspect
from typing import Callable, TypeVar

T = TypeVar("T")


def add_suffix_to_doc(suffix: str) -> Callable[[T], T]:
    """
    Decorator to add a suffix to the docstring of an object.

    Parameters
    ----------
    suffix : str
        The suffix to add.

    Returns
    -------
    Callable[[T], T]
        The decorator that adds the input suffix to the docstring of
        any object.
    """

    def decorator(obj):
        doc = inspect.cleandoc(obj.__doc__ or "")
        if doc:
            obj.__doc__ = doc + "\n\n" + suffix
        else:
            obj.__doc__ = suffix
        return obj

    return decorator
