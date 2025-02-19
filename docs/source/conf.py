# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ClinicaDL"
copyright = "2025, ARAMIS Lab"
author = "ARAMIS Lab"
release = "2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx_design"]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/aramis-lab/clinicadl",
    "repository_branch": "dev",
    "navigation_with_keys": False,
    "show_navbar_depth": 4,
}
html_title = "ClinicaDL documentation"
html_static_path = ["_static"]
