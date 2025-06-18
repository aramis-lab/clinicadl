import inspect
from datetime import date
from pathlib import Path

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
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
]

napoleon_use_admonition_for_references = True
napoleon_use_admonition_for_notes = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
intersphinx_mapping = {
    "torchio": ("https://torchio.readthedocs.io/", None),
    "monai": ("https://docs.monai.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
extlinks = {
    "pathlib.Path": (
        "https://docs.python.org/fr/3.13/library/pathlib.html#concrete-paths/%s",
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
    "torchio": ("https://torchio.readthedocs.io/%s", None),
    "torch": ("https://pytorch.org/docs/stable/%s", None),
    "torchvision": ("https://docs.pytorch.org/vision/main/%s", None),
    "monai": ("https://docs.monai.io/en/stable/%s", None),
    "github": ("https://github.com/aramis-lab/clinicadl/%s", None),
    "wikipedia": ("https://en.wikipedia.org/wiki/%s", None),
}
language = "en"
# pygments_style = "friendly"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "backreferences_dir": Path("api", "generated"),  # where mini-galleries are stored
    "doc_module": (
        "clinicadl",
    ),  # generate mini-galleries for all the objects in clinicadl
}

# sphinxcontrib-bibtex
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

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
html_title = f"{project} {version}"

# Add custom css instructions from themes/custom.css
font_awesome = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/"
html_css_files = [
    "custom.css",
    f"{font_awesome}all.min.css",
    f"{font_awesome}fontawesome.min.css",
    f"{font_awesome}solid.min.css",
    f"{font_awesome}brands.min.css",
]
