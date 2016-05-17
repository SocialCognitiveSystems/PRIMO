#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:47:59 2016

@author: jpoeppel
"""

import unittest
import numpy as np
from primo.network import BayesianNetwork
from primo.io import XMLBIFParser
from primo.inference.order import Orderer
from primo.inference.exact import BucketElimination

class EliminationOderTest(unittest.TestCase):
    
    def test_min_degree_elimination_order(self):
        bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        order = Orderer.get_min_degree_order(bn)
        self.assertEqual(order, ["slippery_road", "wet_grass", "sprinkler", "winter", "rain"])
        
        """
            TODO BETTER TEST WITH CERTAIN ORDER!
        """
        #Check error handling
        with self.assertRaises(TypeError) as cm:
            Orderer.get_min_degree_order("Not a Bayesian Network.")
        self.assertEqual(str(cm.exception), "Only Bayesian Networks are currently supported.")
        
    def test_random_elimination_order(self):
        bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        order = Orderer.get_random_order(bn)
        variables = ["slippery_road", "winter", "rain", "sprinkler", "wet_grass"]
        self.assertEqual(len(order), len(variables))
        for v in variables:
            self.assertTrue(v in order)
            
        #Check error handling
        with self.assertRaises(TypeError) as cm:
            Orderer.get_min_degree_order("Not a Bayesian Network.")
        self.assertEqual(str(cm.exception), "Only Bayesian Networks are currently supported.")

class VariableEliminationTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
    
    def test_marginal(self):
        resFactor = BucketElimination.marginals(self.bn, ["winter"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.6, 0.4]))
        
        
#    def test_marginal_evidence(self):
#        resFactor = BucketElimination.marginals(self.bn, ["winter"])
#        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.6, 0.4]))
        
        
class FactorEliminationTest(unittest.TestCase):
#    
#    def test_marginal(self):
#        self.fail("TODO")
#        
#    def test_marginal_evidence(self):
#        self.fail("TODO")
    pass
        
        
if __name__ == "__main__":
    #Workaround so that this script also finds the resource files when run directly
    # from within the tests folder
    import os
    os.chdir("../..")
    unittest.main()