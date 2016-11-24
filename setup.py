#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:18:15 2016
Setup skript for installing primo and it's dependencies.
@author: jpoeppel
"""

from setuptools import setup

if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'")
    print()

version = 0.5

setup(name="primo",
      version=version,
      description="(Partial) reimplementation of PRobabilistic Inference MOdules",
      long_description="This project is a (partial) reimplementation of the original " \
                       "probabilistic inference modules which can be found at " \
                       "https://github.com/mbaumBielefeld/PRIMO. This reimplementation "\
                       "follows the same general idea, but restructured and unified the "\
                       "underlying datatypes to allow a more concise API and more efficient "\
                       "manipulation, e.g. by the inference algorithm. In turn the inference "\
                       "algorithms have been rewritten and partly extended. For most if not "\
                       "all use cases this implementation should be easier to use and more "\
                       "performant than the original.",
      author="Jan Pöppel",
      author_email="jpoeppel@techfak.uni-bielefeld.de",
      packages = ["primo"],
      install_requires=["numpy", "networkx", "lxml"])


distutils.core.setup(
    name='primo',
    version='1.0',
    description='PRobabilistic Inference MOdules',
    long_description='This project contains probabilistic inference modules for Python. Our aim is to create a library which offers well known probabilistic (graphical) models like Bayesian or temporal networks. A variety of inference algorithms will be implemented soon.',
    author='Manuel Baum, Hendrik Buschmeier, Denis John, Lukas Kettenbach, Max Koch',
    url='https://github.com/mbaumBielefeld/PRIMO',
    download_url='https://github.com/mbaumBielefeld/PRIMO/archive/develop.zip',
    packages = ['primo', 'primo.inference']
)