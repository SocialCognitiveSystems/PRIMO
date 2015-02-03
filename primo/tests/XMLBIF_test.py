#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of PRIMO -- Probabilistic Inference Modules.
# Copyright (C) 2013-2015 Social Cognitive Systems Group, 
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

import os
import unittest

import numpy

from primo.io import XMLBIF
from primo.networks import BayesianNetwork
from primo.nodes import DiscreteNode


class ImportExportTest(unittest.TestCase):
    def setUp(self):
        # Create BayesNet
        self.bn = BayesianNetwork();
        # Create Nodes
        weather0 = DiscreteNode("Weather0", ["Sun", "Rain"])
        weather = DiscreteNode("Weather", ["Sun", "Rain"])
        ice_cream_eaten = DiscreteNode("Ice Cream Eaten", [True, False])
        # Add nodes
        self.bn.add_node(weather0)
        self.bn.add_node(weather)
        self.bn.add_node(ice_cream_eaten)
        # Add edges
        self.bn.add_edge(weather, ice_cream_eaten)
        self.bn.add_edge(weather0, weather);
        # Set probabilities
        cpt_weather0 = numpy.array([.6, .4])
        weather0.set_probability_table(cpt_weather0, [weather0])
        cpt_weather = numpy.array([[.7, .5],
                                   [.3, .5]])
        weather.set_probability_table(cpt_weather, [weather0, weather])
        ice_cream_eaten.set_probability(.9, [(ice_cream_eaten, True), (weather, "Sun")])
        ice_cream_eaten.set_probability(.1, [(ice_cream_eaten, False), (weather, "Sun")])
        ice_cream_eaten.set_probability(.2, [(ice_cream_eaten, True), (weather, "Rain")])
        ice_cream_eaten.set_probability(.8, [(ice_cream_eaten, False), (weather, "Rain")])
    
    def test_import_export(self):
        # write BN 
        xmlbif = XMLBIF(self.bn, "Test Net")
        xmlbif.write("test_out.xmlbif")
        # read BN
        bn2 = XMLBIF.read("test_out.xmlbif")
        for node1 in self.bn.get_nodes():
            name_found = False
            cpd_equal = False
            value_range_equal = False
            str_equal = False
            pos_equal = False
            for node2 in bn2.get_nodes():
                print(node2.name)
                print(node2.get_cpd())
                # Test node names
                print(node2.name)
                if node1.name == node2.name:
                    name_found = True
                    cpd_equal = node1.get_cpd() == node2.get_cpd()
                    value_range_equal = node1.get_value_range() == node2.get_value_range()
                    str_equal = str(node1) == str(node2)
                    pos_equal = node1.pos == node2.pos
            self.assertTrue(name_found)
            self.assertTrue(cpd_equal)
            self.assertTrue(value_range_equal)
            self.assertTrue(str_equal)
            self.assertTrue(pos_equal)
        # remove file
        os.remove("test_out.xmlbif")
        

#include this so you can run this test without nose
if __name__ == '__main__':
    unittest.main()
