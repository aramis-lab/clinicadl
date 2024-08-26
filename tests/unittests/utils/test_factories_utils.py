from clinicadl.utils.factories import get_args_and_defaults


def test_get_default_args():
    def f(a, b="b", c=0, d=None):
        return None

    args, defaults = get_args_and_defaults(f)
    assert args == ["a", "b", "c", "d"]
    assert defaults == {"b": "b", "c": 0, "d": None}
