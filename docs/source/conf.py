# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Import standard libs
import sys
from pathlib import Path

cwd = Path.cwd()
ROOT_PATH = str(cwd.parent.parent.parent)
sys.path.insert(0, ROOT_PATH)

import simpml


# -- Project information -----------------------------------------------------

project = "SimpML"
copyright = "2023, NI Corp."
author = "NI Corp."
# TODO: Get the version from within the `optimalai` package
version = "0.1"
release = version
language = "en"
html_logo = 'images/SimpML white.png'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_rtd_size"
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------

nbsphinx_allow_errors = True

# Use this kernel instead of your default kernel
nbsphinx_kernel_name = 'python3'

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = 'always'

# Use this option to include require.js
nbsphinx_requirejs_path = 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme'
sphinx_rtd_size_width = "100%"
html_theme_options = {
    "style_nav_header_background": "#03B484"
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
