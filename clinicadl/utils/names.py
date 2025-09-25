import re


def camel_to_snake(name: str) -> str:
    # Step 1: insert underscore between acronym and normal word
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # Step 2: insert underscore between lowercase/digit and uppercase
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    # Step 3: insert underscore between letters and digits
    s3 = re.sub(r"([a-zA-Z])([0-9])", r"\1_\2", s2)
    return s3.lower()
