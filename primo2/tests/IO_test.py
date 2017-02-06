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


import unittest
import numpy as np

from primo2.network import BayesianNetwork
from primo2.io import XMLBIFParser
from primo2.nodes import DiscreteNode

class XMLBIFTest(unittest.TestCase):
    

    def test_readXMLBIF(self):
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        
        nodes = bn.get_all_nodes()        
        self.assertTrue("slippery_road" in nodes)
        self.assertTrue("sprinkler" in nodes)
        self.assertTrue("rain" in nodes)
        self.assertTrue("wet_grass" in nodes)
        self.assertTrue("winter" in nodes)
        self.assertEqual(len(nodes), 5)
        slipperyNode = bn.get_node("slippery_road")
        self.assertTrue("rain" in slipperyNode.parents)
        sprinklerNode = bn.get_node("sprinkler")
        self.assertTrue("winter" in sprinklerNode.parents)
        rainNode = bn.get_node("rain")
        self.assertTrue("winter" in rainNode.parents)
        cpt = np.array([[0.8,0.1],[0.2,0.9]])        
        np.testing.assert_array_almost_equal(rainNode.cpd, cpt)
        
        wetNode = bn.get_node("wet_grass")
        self.assertTrue("sprinkler" in wetNode.parents)
        self.assertTrue("rain" in wetNode.parents)
        self.assertTrue("true" in wetNode.values)
        cpt = np.array([[[0.95, 0.8],[0.1,0.0]], [[0.05, 0.2],[0.9, 1.0]]])
        self.assertEqual(wetNode.get_probability("false", {"rain":["true"], "sprinkler":["false"]}),0.2)
        self.assertEqual(wetNode.get_probability("true", {"rain":["false"], "sprinkler":["true"]}),0.1)
        
#        np.testing.assert_array_almost_equal(wetNode.cpd, cpt)
        
    def test_readXMLBIF_different_parent_sizes(self):
        bn = XMLBIFParser.parse("primo2/tests/testfile.xbif")
#        nodes = bn.get_all_nodes()
        johnNode = bn.get_node("John_calls")
        cpt = np.array([[[0.8,0.5,0.7],[0.6,0.2,0.1]],[[0.2,0.5,0.3],[0.4,0.8,0.9]]])
        np.testing.assert_array_almost_equal(johnNode.cpd, cpt)
        
    def test_writeXMLBIF_simple(self):
        path= "primo2/tests/test.xbif"
        bn = BayesianNetwork()
        n1 = DiscreteNode("a")
        n2 = DiscreteNode("b", ["happy", "sad"])
        bn.add_node(n1)
        bn.add_node(n2)
        bn.add_edge(n1,n2)
        XMLBIFParser.write(bn, path)

        bn2 = XMLBIFParser.parse(path)
        for n in bn2.get_all_nodes():
            tmpn = bn.get_node(n)
            for value in tmpn.values:
                self.assertTrue(value in n.values)
            for p in tmpn.parents.keys():
                self.assertTrue(p in n.parents)
            np.testing.assert_almost_equal(tmpn.cpd, n.cpd)
        # remove testfile
        import os
        os.remove(path)       
            
    def test_writeXMLBIF(self):
        testPath = "primo2/tests/testSlippery.xbif"
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        XMLBIFParser.write(bn, testPath)
        bn2 = XMLBIFParser.parse(testPath)
        for n in bn2.get_all_nodes():
            tmpn = bn.get_node(n)
            for value in tmpn.values:
                self.assertTrue(value in n.values)
            for i,p in enumerate(tmpn.parentOrder):
                self.assertEqual(p, n.parentOrder[i])
            np.testing.assert_almost_equal(tmpn.cpd, n.cpd)
        # remove testfile
        import os
        os.remove(testPath)
        
if __name__ == "__main__":
    #Workaround so that this script also finds the resource files when run directly
    # from within the tests folder
    import os
    os.chdir("../..")
    unittest.main()