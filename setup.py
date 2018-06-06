#! /usr/bin/env python
#
# Copyright (C) 2012-2018 Michael Waskom
import os
# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"] = "."

DESCRIPTION = "seaborn: statistical data visualization"
LONG_DESCRIPTION = """\
Seaborn is a library for making attractive and informative statistical graphics in Python. It is built on top of matplotlib and tightly integrated with the PyData stack, including support for numpy and pandas data structures and statistical routines from scipy and statsmodels.

Some of the features that seaborn offers are

- Several built-in themes that improve on the default matplotlib aesthetics
- Tools for choosing color palettes to make beautiful plots that reveal patterns in your data
- Functions for visualizing univariate and bivariate distributions or for comparing them between subsets of data
- Tools that fit and visualize linear regression models for different kinds of independent and dependent variables
- Functions that visualize matrices of data and use clustering algorithms to discover structure in those matrices
- A function to plot statistical timeseries data with flexible estimation and representation of uncertainty around the estimate
- High-level abstractions for structuring grids of plots that let you easily build complex visualizations
"""

DISTNAME = 'seaborn'
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@nyu.edu'
URL = 'https://seaborn.pydata.org'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mwaskom/seaborn/'
VERSION = '0.9.dev'

INSTALL_REQUIRES = [
    'numpy>=1.9.3',
    'scipy>=0.14.0',
    'pandas>=0.15.2',
    'matplotlib>=1.4.3',
]

PACKAGES = [
    'seaborn',
    'seaborn.colors',
    'seaborn.external',
    'seaborn.tests',
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'License :: OSI Approved :: BSD License',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Multimedia :: Graphics',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS'
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS
    )
