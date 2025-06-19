from clinicadl.utils.factories import get_args_from, get_defaults_from


def f(a, b="b", c=0, d=None):
    return None


def test_get_defaults_from():
    defaults = get_defaults_from(f)
    assert defaults == {"b": "b", "c": 0, "d": None}


def test_get_args_from():
    args = get_args_from(f)
    assert args == ["a", "b", "c", "d"]
