#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:16:46 2016

@author: jpoeppel
"""


import unittest
import numpy as np

from primo.network import BayesianNetwork
from primo.io import XMLBIFParser

class XMLBIFTest(unittest.TestCase):
    

    def test_readXMLBIF(self):
        bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        
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
        bn = XMLBIFParser.parse("primo/tests/testfile.xbif")
#        nodes = bn.get_all_nodes()
        johnNode = bn.get_node("John_calls")
        cpt = np.array([[[0.8,0.5,0.7],[0.6,0.2,0.1]],[[0.2,0.5,0.3],[0.4,0.8,0.9]]])
        np.testing.assert_array_almost_equal(johnNode.cpd, cpt)
        
#    def test_writeXMLBIF(self):
#        self.fail("TODO")
        
if __name__ == "__main__":
    #Workaround so that this script also finds the resource files when run directly
    # from within the tests folder
    import os
    os.chdir("../..")
    unittest.main()