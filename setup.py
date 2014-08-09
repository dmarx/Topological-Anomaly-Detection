#!/usr/bin/env python

import sys
import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()

setup(name='tad',
      version='0.1',
      description='Topological Anomaly Detection (TAD)',
      long_description=README,
      author='David Marx',
      #~ author_email='',
      url='https://github.com/dmarx/Topological-Anomaly-Detection',
      packages=['tad'],
      package_dir = {'': "src"},include_package_data=True,
      zip_safe=False,
      license='BSD-3',
      install_requires=['networkx', 'numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn'],
      entry_points={
        'console_scripts':
            ['tad_demo=tad.demo']
      }
     )


