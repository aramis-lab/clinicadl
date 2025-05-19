import inspect
from datetime import date

import clinicadl

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ClinicaDL"
author = "ARAMIS Lab"
copyright = f"{date.today().year}, {author}"
# version = release = clinicadl.__version__
version = "2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "torchio": ("https://torchio.readthedocs.io", None),
    "monai": ("https://docs.monai.io/en/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}
extlinks = {
    "pathlib.Path": (
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
# pygments_style = "friendly"

# -- Hide function with @overload ---------------------------------------

typehints_use_signature = True  # replaces the signature with type hints
typehints_use_signature_return = True
typehints_document_rtype = False


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


def setup(app):
    app.connect("autodoc-skip-member", skip_overload_members)


# -- Hide function with @overload ---------------------------------------

typehints_use_signature = True  # replaces the signature with type hints


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


def setup(app):
    app.connect("autodoc-skip-member", skip_overload_members)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

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
