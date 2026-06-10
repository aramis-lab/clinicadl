from typing import Any, Union, get_args, get_origin


def is_dict_type(typing: Any) -> bool:
    """
    Returns True if `tp` is a dict or contains dict inside a Union/Optional.
    """
    origin = get_origin(typing)
    if origin is dict:
        return True
    if origin is Union:
        return any(is_dict_type(arg) for arg in get_args(typing))
    return False
