from glob import glob


def insensitive_glob(pattern_glob: str, recursive: bool = False) -> list[str]:
    """
    Perform a case-insensitive glob search.

    Args:
        pattern_glob (str): The pattern to search for, sensitive to case.
        recursive (bool, optional): If True, performs the glob search recursively. Default is False.

    Returns:
        List[str]: A list of matching file paths, case-insensitive to the given pattern.
    """

    insensitive_pattern = "".join(map(_make_case_insensitive_pattern, pattern_glob))
    return glob(insensitive_pattern, recursive=recursive)


def _make_case_insensitive_pattern(c: str) -> str:
    """
    Converts a character to a case-insensitive pattern for glob matching.
    """
    return "[%s%s]" % (c.lower(), c.upper()) if c.isalpha() else c
