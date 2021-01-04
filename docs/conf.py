# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
#autodoc_mock_imports = ['sphinx_bootstrap_theme']

#import sphinx_bootstrap_theme

import os
import sys
sys.path.insert(0, os.path.abspath('../pyAudioDspTools'))


# -- Project information -----------------------------------------------------

project = 'pyAudioDspTools'
copyright = '2020, Arjaan Auinger'
author = 'Arjaan Auinger'

# The full version, including alpha/beta/rc tags
release = '0.7.9'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.coverage'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'classic'

colour = "Ivory"

html_theme_options = {
    "bgcolor": colour,
    "relbarbgcolor": colour,
    "sidebarbgcolor": "DimGrey",
    "sidebarlinkcolor": "Cornsilk"
}

#html_theme = 'sphinx_rtd_theme'
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '../SmallLogo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'
