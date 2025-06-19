import enum
import inspect
import typing
from datetime import date
from typing import Annotated, Union, get_args, get_origin

from pydantic import BaseModel

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ClinicaDL"
author = "ARAMIS Lab"
copyright = f"{date.today().year}, {author}"
version = "2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_member_order = "bysource"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torchio": ("https://torchio.readthedocs.io", None),
    "monai": ("https://docs.monai.io/en/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "torchvision": ("https://pytorch.org/vision/main", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
extlinks = {
    "pathlib.Path": (
        "https://docs.python.org/fr/3.13/library/pathlib.html#concrete-paths%s",
        None,
    ),
    "Path": (
        "https://docs.python.org/fr/3.13/library/pathlib.html#concrete-paths%s",
        None,
    ),
    "tutorials": (
        "https://github.com/aramis-lab/clinicadl-tutorials/tree/main/%s",
        None,
    ),
    "zoo": (
        "https://github.com/aramis-lab/clinicadl-zoo/tree/main/%s",
        None,
    ),
}
language = "en"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

typehints_use_signature = True  # replaces the signature with type hints
typehints_use_signature_return = True
typehints_document_rtype = False

html_theme = "furo"
html_theme_options = {
    "light_logo": "black_logo.png",
    "dark_logo": "white_logo.png",
}

html_static_path = ["_static"]
html_favicon = "_static/black_logo.png"
html_copy_source = False
html_show_sourcelink = False
html_title = f"{project} {version} documentation"

autodoc_typehints = "signature"


# -- Hide function with @overload ---------------------------------------
def is_overload_function(obj):
    # `overload` sets __code__.co_code to b''
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        try:
            return obj.__code__.co_code == b""
        except AttributeError:
            return False
    return False


def skip_overload_members(app, what, name, obj, skip, options):
    if is_overload_function(obj):
        return True  # skip this member
    return None


def skip_private_members(app, what, name, obj, skip, options):
    if name.startswith("_"):
        return True
    return skip


# -- Simplify type hints for Pydantic models --------------------------------


ReversePydanticTypes = {
    "int ≥ 0": "PositiveInt",
    "int > 0": "NonNegativeInt",
    "int ≤ 0": "NegativeInt",
    "int < 0": "NonPositiveInt",
    "float ≥ 0": "PositiveFloat",
    "float > 0": "NonNegativeFloat",
    "float ≤ 0": "NegativeFloat",
    "float < 0": "NonPositiveFloat",
    # you can add other types here
}


def simplify_type(tp):
    origin = get_origin(tp)

    # Handle Union
    if origin is Union:
        return " | ".join(simplify_type(arg) for arg in get_args(tp))

    # Handle Annotated (Pydantic constraints)
    if origin is Annotated:
        base, *constraints = get_args(tp)
        # otherwise, fallback with explicit constraints
        parts = base.__name__ if hasattr(base, "__name__") else str(base)
        for constraint in constraints:
            if hasattr(constraint, "ge"):
                parts += f" ≥ {constraint.ge}"
            if hasattr(constraint, "gt"):
                parts += f" > {constraint.gt}"
            if hasattr(constraint, "le"):
                parts += f" ≤ {constraint.le}"
            if hasattr(constraint, "lt"):
                parts += f" < {constraint.lt}"

        if parts in ReversePydanticTypes:
            type_ = ReversePydanticTypes.get(parts, parts)
            return type_
        return parts

    # Handle Literal
    if origin is typing.Literal:
        values = get_args(tp)
        return f"Literal[{', '.join(repr(v) for v in values)}]"

    # Handle Tuple
    if origin in (tuple, typing.Tuple):
        inner = ", ".join(simplify_type(arg) for arg in get_args(tp))
        return f"tuple[{inner}]"

    # Handle Tuple
    if origin in (dict, typing.Dict):
        inner = ", ".join(simplify_type(arg) for arg in get_args(tp))
        return f"dict[{inner}]"

    # Handle Enum
    if inspect.isclass(tp) and issubclass(tp, enum.Enum):
        values = ", ".join([f'"{e.value}"' for e in tp])
        return f"{tp.__name__} ({values})"

    # Handle List
    if origin in (list, typing.List):
        inner = ", ".join(simplify_type(arg) for arg in get_args(tp))
        return f"list[{inner}]"

    # Base case
    if hasattr(tp, "__name__"):
        return tp.__name__

    return str(tp)


def rewrite_class_signature(
    app, what, name, obj, options, signature, return_annotation
):
    print(obj)
    if not isinstance(obj, type) or not issubclass(obj, BaseModel):
        return

    try:
        annots = typing.get_type_hints(obj, include_extras=True)
    except Exception as e:
        print(f"[ERROR] Failed to get type hints for {name}: {e}")
        return

    parts = []

    for field_name, field_type in annots.items():
        simplified = simplify_type(field_type)
        field = obj.model_fields[field_name]
        if field.default is not None and field.default != ...:
            default = repr(field.default)
            part = f"{field_name}: {simplified} = {default}"
        else:
            part = f"{field_name}: {simplified}"
        parts.append(part)

    new_sig = f"({', '.join(parts)})"
    return new_sig, None


def setup(app):
    app.connect("autodoc-skip-member", skip_overload_members)
    app.connect("autodoc-skip-member", skip_private_members)
    app.connect("autodoc-process-signature", rewrite_class_signature)
