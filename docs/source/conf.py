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
import os
import sys
from subprocess import run

import solar_theme

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = 'DGP'
# pylint: disable=W0622
copyright = '2021, Toyota Research Institute'
author = 'Toyota Research Institute'

# The full version, including alpha/beta/rc tags
cmd = "git describe --tags --match v[0-9]*.[0-9]*"
release = run(cmd.split(), capture_output=True, check=True).stdout.decode().strip()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'm2r2']
source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# Currently, excludes all autogen files
exclude_patterns = ['**/*pb2_grpc.py', '**/*_pb2.py']

# -- Options for HTML output -------------------------------------------------
html_theme = 'solar_theme'
html_theme_path = [solar_theme.theme_path]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'tri_colors.css',
]
html_style = 'tri_colors.css'
html_sidebars = {
    '**': [
        # located at _templates/
        'side.html',
        #auto-gen
        'searchbox.html',
        'links.html',
        'localtoc.html',
        'relations.html',
    ]
}
