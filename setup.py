#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of PRIMO2 -- Probabilistic Inference Modules.
# Copyright (C) 2013-2017 Social Cognitive Systems Group, 
#                         Faculty of Technology, Bielefeld University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as 
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public 
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

import setuptools
import sys

if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'")
    print()

version = 0.91

setuptools.setup(
    name="primo2",
    version=version,
    description="(Partial) reimplementation of PRobabilistic Inference MOdules",
    long_description="This project is a (partial) reimplementation of the original " \
        "probabilistic inference modules which can be found at " \
        "https://github.com/hbuschme/PRIMO. This reimplementation " \
        "follows the same general idea, but restructured and unified the " \
        "underlying datatypes to allow a more concise API and more efficient " \
        "manipulation, e.g. by the inference algorithm. In turn the inference " \
        "algorithms have been rewritten and partly extended. For most if not " \
        "all use cases this implementation should be easier to use and more " \
        "performant than the original.",
    author="Jan Pöppel",
    author_email="jpoeppel@techfak.uni-bielefeld.de",
    packages=[
            "primo2",
            "primo2.inference"
        ],
    install_requires = [
            "lxml",
            "numpy",
            "networkx",
            "six"
        ]
    )
