from clinicadl.utils.doc import add_suffix_to_doc


def test_add_suffix_to_doc():
    @add_suffix_to_doc("A suffix.")
    def _test_func():
        """A docstring."""

    @add_suffix_to_doc("A suffix.")
    def _test_func_empty():
        pass

    @add_suffix_to_doc("A suffix.")
    class _TestClass:
        """
        A

        docstring.
        """

    assert _test_func.__doc__ == "A docstring.\n\nA suffix."
    assert _test_func_empty.__doc__ == "A suffix."
    assert _TestClass.__doc__ == "A\n\ndocstring.\n\nA suffix."
