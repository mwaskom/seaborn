#! /usr/bin/env python
#
# Copyright (C) 2012 Michael Waskom <mwaskom@stanford.edu>

descr = """Seaborn: improved statistical visualization using Matplotlib"""

import os


DISTNAME = 'seaborn'
DESCRIPTION = descr
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@stanford.edu'
URL = 'http://stanford.edu/~mwaskom/software/seaborn/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = ('https://github.com/mwaskom/seaborn/zipball/master'
                '#egg=seaborn=dev')
VERSION = '0.2.1'

from setuptools import setup

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        packages=['seaborn', 'seaborn.tests'],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3.3',
                     'License :: OSI Approved :: BSD License',
                     'Topic :: Scientific/Engineering :: Visualization',
                     'Topic :: Multimedia :: Graphics',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        install_requires=["husl", "moss>0.1", "patsy",
                          "pandas", "statsmodels", "six"],
          )
