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
import time
import seaborn
from seaborn._core.properties import PROPERTIES

sys.path.insert(0, os.path.abspath('sphinxext'))


# -- Project information -----------------------------------------------------

project = 'seaborn'
copyright = f'2012-{time.strftime("%Y")}'
author = 'Michael Waskom'
version = release = seaborn.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (amed 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'matplotlib.sphinxext.plot_directive',
    'gallery_generator',
    'tutorial_builder',
    'numpydoc',
    'sphinx_copybutton',
    'sphinx_issues',
    'sphinx_design',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The root document.
root_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'docstrings', 'nextgen', 'Thumbs.db', '.DS_Store']

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = 'literal'

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# Sphinx-issues configuration
issues_github_path = 'mwaskom/seaborn'

# Include the example source for plots in API docs
plot_include_source = True
plot_formats = [('png', 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

# Don't add a source link in the sidebar
html_show_sourcelink = False

# Control the appearance of type hints
autodoc_typehints = "none"
autodoc_typehints_format = "short"

# Allow shorthand references for main function interface
rst_prolog = """
.. currentmodule:: seaborn
"""

# Define replacements (used in whatsnew bullets)
rst_epilog = """

.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |API| replace:: :raw-html:`<span class="badge badge-api">API</span>` :raw-latex:`{\small\sc [API]}`
.. |Defaults| replace:: :raw-html:`<span class="badge badge-defaults">Defaults</span>` :raw-latex:`{\small\sc [Defaults]}`
.. |Docs| replace:: :raw-html:`<span class="badge badge-docs">Docs</span>` :raw-latex:`{\small\sc [Docs]}`
.. |Feature| replace:: :raw-html:`<span class="badge badge-feature">Feature</span>` :raw-latex:`{\small\sc [Feature]}`
.. |Enhancement| replace:: :raw-html:`<span class="badge badge-enhancement">Enhancement</span>` :raw-latex:`{\small\sc [Enhancement]}`
.. |Fix| replace:: :raw-html:`<span class="badge badge-fix">Fix</span>` :raw-latex:`{\small\sc [Fix]}`
.. |Build| replace:: :raw-html:`<span class="badge badge-build">Build</span>` :raw-latex:`{\small\sc [Deps]}`

"""  # noqa

rst_epilog += "\n".join([
    f".. |{key}| replace:: :ref:`{key} <{val.__class__.__name__.lower()}_property>`"
    for key, val in PROPERTIES.items()
])

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ['_static', 'example_thumbs']
for path in html_static_path:
    if not os.path.exists(path):
        os.makedirs(path)

html_css_files = [f'css/custom.css?v={seaborn.__version__}']

html_logo = "_static/logo-wide-lightbg.svg"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mwaskom/seaborn",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "StackOverflow",
            "url": "https://stackoverflow.com/tags/seaborn",
            "icon": "fab fa-stack-overflow",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/michaelwaskom",
            "icon": "fab fa-twitter",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": False,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "header_links_before_dropdown": 8,
}

html_context = {
    "default_mode": "light",
}

html_sidebars = {
    "index": [],
    "examples/index": [],
    "**": ["sidebar-nav-bs.html"],
}

# -- Intersphinx ------------------------------------------------

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None)
}
