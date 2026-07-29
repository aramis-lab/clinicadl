import inspect
import re
import sys
from datetime import date
from pathlib import Path
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
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
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
    "torchvision": ("https://pytorch.org/vision/stable", None),
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
    "monai": ("https://monai.readthedocs.io/en/stable/%s", None),
    "github": ("https://github.com/aramis-lab/clinicadl/%s", None),
    "wikipedia": ("https://en.wikipedia.org/wiki/%s", None),
    "bids": ("https://bids-specification.readthedocs.io/en/stable/%s", None),
    "nibabel": ("https://nipy.org/nibabel/%s", None),
    "clinica": ("https://aramislab.paris.inria.fr/clinica/docs/public/latest/%s", None),
}
language = "en"
# pygments_style = "friendly"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "backreferences_dir": Path("generated"),  # where mini-galleries are stored
    "doc_module": (
        "clinicadl",
    ),  # generate mini-galleries for all the objects in clinicadl
    "download_all_examples": False,  # disabling download button of all scripts
}

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


# -- Hide methods/validators on Pydantic models -------------------------
# Config pages use `:inherited-members: BaseModel` to surface inherited
# *fields* (e.g. `channels` on ConvEncoderConfig, defined in a mix-in).
# That option also drags in inherited methods and validators, which we
# don't want on config pages. Fields are not routines, so we only need to
# drop members that are functions/methods defined on a Pydantic model.
def _owner_class(obj):
    """Return the class on which a routine is defined, else None."""
    func = getattr(obj, "__func__", obj)  # unwrap classmethod/staticmethod
    if not (inspect.isfunction(func) or inspect.ismethod(func)):
        return None
    qualname = getattr(func, "__qualname__", "")
    if "." not in qualname:
        return None
    owner = sys.modules.get(func.__module__)
    for part in qualname.split(".")[:-1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    return owner if inspect.isclass(owner) else None


def skip_pydantic_methods(app, what, name, obj, skip, options):
    from pydantic import BaseModel

    owner = _owner_class(obj)
    if owner is not None and issubclass(owner, BaseModel):
        return True  # drop methods and validators, keep fields
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


# -- Generate "What's new?" pages from CHANGELOG.md -------------------------
# Each release section in the root CHANGELOG.md ("## [version] - date") is
# turned into its own Markdown page under docs/whats_new/, rendered natively
# by myst-parser. whats_new.rst globs that folder; a zero-padded numeric
# prefix on each file keeps the toctree in CHANGELOG order (newest first).
_DOCS_DIR = Path(__file__).parent
_CHANGELOG = _DOCS_DIR.parent / "CHANGELOG.md"
_WHATS_NEW_DIR = _DOCS_DIR / "whats_new"
# matches e.g. "## [2.0.0] – 2026-06-10" (en-dash, em-dash or hyphen)
_RELEASE_RE = re.compile(r"^##\s+\[(?P<version>[^\]]+)\]\s*[–—-]\s*(?P<date>.+?)\s*$")


def _split_changelog(text):
    """Yield (version, date, body) tuples, one per release, in file order."""
    current = None
    lines = []
    for line in text.splitlines():
        match = _RELEASE_RE.match(line)
        if match:
            if current is not None:
                yield current[0], current[1], "\n".join(lines).strip()
            current = (match["version"], match["date"])
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        yield current[0], current[1], "\n".join(lines).strip()


def _promote_headings(body):
    """Promote sub-section headings by one level so the page h1 is unique."""
    return re.sub(r"^(#{2,})\s", lambda m: m.group(1)[1:] + " ", body, flags=re.M)


def generate_whats_new(app=None):
    if not _CHANGELOG.exists():
        return
    releases = list(_split_changelog(_CHANGELOG.read_text(encoding="utf-8")))
    _WHATS_NEW_DIR.mkdir(exist_ok=True)

    for page in _WHATS_NEW_DIR.glob("*.md"):
        page.unlink()

    width = max(3, len(str(len(releases))))
    for idx, (version, date_str, body) in enumerate(releases):
        page = f"# {version}\n\n*Released {date_str}*\n\n{_promote_headings(body)}\n"
        release_id = re.sub(r"[^0-9A-Za-z.]+", "-", version)
        (_WHATS_NEW_DIR / f"{idx:0{width}d}-release-{release_id}.md").write_text(
            page, encoding="utf-8"
        )


def setup(app):
    app.connect("autodoc-skip-member", skip_overload_members)
    app.connect("autodoc-skip-member", skip_pydantic_methods)
    app.connect("config-inited", lambda app, config: generate_whats_new(app))
