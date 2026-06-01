import inspect
from datetime import date
from typing import Annotated as AnnotatedAlias
from typing import get_args, get_origin

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
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    # "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxcontrib.autodoc_pydantic",
]

autodoc_pydantic_model_show_json = False

napoleon_use_admonition_for_references = True
napoleon_use_admonition_for_notes = True
napoleon_numpy_docstring = True

napoleon_custom_sections = [("Returns", "params_style"), ("Attributes", "params_style")]


templates_path = ["_templates"]
exclude_patterns = []
autodoc_member_order = "bysource"

autodoc_typehints = "description"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torchio": ("https://torchio.readthedocs.io", None),
    "monai": ("https://monai.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "torchvision": ("https://pytorch.org/vision/main", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "clinicadl": ("https://clinicadl.readthedocs.io/en/latest/", None),
}
extlinks = {
    "pathlib.Path": (
        "https://docs.python.org/fr/3/library/pathlib.html#concrete-paths%s",
        None,
    ),
    "Path": (
        "https://docs.python.org/fr/3/library/pathlib.html#concrete-paths%s",
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
    "bids": ("https://bids-specification.readthedocs.io/en/stable/%s", None),
    "clinica": ("https://aramislab.paris.inria.fr/clinica/docs/public/latest/%s", None),
}
language = "en"
# pygments_style = "friendly"

# sphinx_gallery_conf = {
#     "examples_dirs": "../examples",  # path to scripts
#     "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
#     "backreferences_dir": Path("generated"),  # where mini-galleries are stored
#     "doc_module": (
#         "clinicadl",
#     ),  # generate mini-galleries for all the objects in clinicadl
# }

# sphinxcontrib-bibtex
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

typehints_use_signature = True  # replaces the signature with type hints
typehints_use_signature_return = True
typehints_document_rtype = False

html_theme = "furo"
html_theme_options = {
    "light_logo": "logos/black_logo.png",
    "dark_logo": "logos/white_logo.png",
}

html_static_path = ["_static"]
html_favicon = "_static/logos/black_logo.png"
html_copy_source = False
html_show_sourcelink = False
html_title = f"{project}"

autodoc_typehints = "signature"

# Add custom css instructions from themes/custom.css
font_awesome = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/"
html_css_files = [
    "custom.css",
    f"{font_awesome}all.min.css",
    f"{font_awesome}fontawesome.min.css",
    f"{font_awesome}solid.min.css",
    f"{font_awesome}brands.min.css",
]


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


# -- Simplify type hints for Pydantic models --------------------------------
_PYDANTIC_TYPE_NAMES = {
    (float, "gt", 0): "PositiveFloat",
    (float, "ge", 0): "NonNegativeFloat",
    (int, "gt", 0): "PositiveInt",
    (int, "ge", 0): "NonNegativeInt",
    (float, "lt", 0): "NegativeFloat",
    (float, "le", 0): "NonPositiveFloat",
}


def typehints_formatter(annotation, config):
    """Replace verbose Pydantic Annotated types with readable names."""
    if get_origin(annotation) is AnnotatedAlias:
        base, *metadata = get_args(annotation)
        for m in metadata:
            for (base_type, attr, value), name in _PYDANTIC_TYPE_NAMES.items():
                if base is base_type and getattr(m, attr, None) == value:
                    return f":py:data:`~pydantic.{name}`"
    return None  # fall back to default


def setup(app):
    app.connect("autodoc-skip-member", skip_overload_members)
