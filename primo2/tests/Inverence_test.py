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
from primo2.networks import BayesianNetwork
from primo2.nodes import DiscreteNode
from primo2.io import XMLBIFParser
from primo2.inference.order import Orderer
from primo2.inference.exact import VariableElimination
from primo2.inference.exact import FactorTree

class EliminationOderTest(unittest.TestCase):
    
    def test_min_degree_elimination_order(self):
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        order = Orderer.get_min_degree_order(bn)
        #Test for all possible/equivalent orders since the actual order might is not
        #determined based on the random nature hash in Python3
        potentialOrders = [["slippery_road", "wet_grass", "sprinkler", "winter", "rain"], 
                           ["slippery_road", "wet_grass", "sprinkler", "rain", "winter"], 
                           ["slippery_road", "wet_grass", "rain", "sprinkler", "winter"],  
                           ["slippery_road", "wet_grass", "rain", "winter", "sprinkler"],
                           ["slippery_road", "wet_grass", "winter", "rain", "sprinkler"],
                           ["slippery_road", "wet_grass", "winter", "sprinkler", "rain"],             
                           ["slippery_road", "winter", "sprinkler", "wet_grass", "rain"],
                           ["slippery_road", "winter", "sprinkler", "rain", "wet_grass"], 
                           ["slippery_road", "winter", "rain", "sprinkler", "wet_grass"], 
                           ["slippery_road", "winter", "rain", "wet_grass", "sprinkler"], 
                           ["slippery_road", "winter", "wet_grass", "sprinkler", "rain"], 
                           ["slippery_road", "winter", "wet_grass", "rain", "sprinkler"],  
                           ["slippery_road", "sprinkler", "winter", "wet_grass", "rain"], 
                           ["slippery_road", "sprinkler", "winter", "rain", "wet_grass"],   
                           ["slippery_road", "sprinkler", "wet_grass", "winter", "rain"],
                           ["slippery_road", "sprinkler", "wet_grass", "rain", "winter"],   
                           ["slippery_road", "sprinkler", "rain", "winter", "wet_grass"],
                           ["slippery_road", "sprinkler", "rain", "wet_grass", "winter"], 
                           ["slippery_road", "rain", "wet_grass", "sprinkler", "winter"], 
                           ["slippery_road", "rain", "wet_grass", "winter", "sprinkler"],
                           ["slippery_road", "rain", "winter", "wet_grass", "sprinkler"], 
                           ["slippery_road", "rain", "winter", "sprinkler", "wet_grass"],
                           ["slippery_road", "rain", "sprinkler", "wet_grass", "winter"], 
                           ["slippery_road", "rain", "sprinkler", "winter", "wet_grass"]]
        self.assertTrue(order in potentialOrders)                   
        
        """
            TODO BETTER TEST WITH CERTAIN ORDER!
        """
        #Check error handling
        with self.assertRaises(TypeError) as cm:
            Orderer.get_min_degree_order("Not a Bayesian Network.")
        self.assertEqual(str(cm.exception), "Only Bayesian Networks are currently supported.")
        
    def test_random_elimination_order(self):
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
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
        self.bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        
    def test_empty_cpt(self):
        bn = BayesianNetwork()
        from primo2.nodes import DiscreteNode
        n1 = DiscreteNode("a")
        n2 = DiscreteNode("b")
        bn.add_node(n1)
        bn.add_node(n2)
        bn.add_edge(n1,n2)
        res = VariableElimination.naive_marginals(bn, ["a"])
        np.testing.assert_array_almost_equal(res.get_potential(), np.array([0.0, 0.0]))
        
    def test_naive_marginals(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["winter"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.6, 0.4]))
        
    def test_naive_marginal_evidence_trivial(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["rain"], {"winter": "true"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.8, 0.2]))
        
    def test_naive_marginal_evidence_trivial_multiple_evidence(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["wet_grass"], {"sprinkler": "true", "rain": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.1, 0.9]))
        
    def test_naive_marginal_evidence(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["wet_grass"], {"winter": "true"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.668, 0.332]))
        
    def test_naive_marginal_evidence_multiple_evidence(self):
        resFactor = VariableElimination.naive_marginals(self.bn, ["wet_grass"], {"winter": "true", "rain": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.02, 0.98]))
        

        
    def test_bucket_marginals(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["winter"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.6, 0.4]))
#        
    def test_bucket_marginal_evidence_trivial(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["rain"], {"wet_grass": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.158858, 0.841142]))
        
    def test_bucket_marginal_evidence_trivial_multiple_evidence(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["wet_grass"], {"sprinkler": "true", "rain": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.1, 0.9]))
    
        
    def test_bucket_marginal_evidence(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["wet_grass"], {"winter": "true"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.668, 0.332]))
        
    def test_bucket_marginal_evidence_multiple_evidence(self):
        resFactor = VariableElimination.bucket_marginals(self.bn, ["wet_grass"], {"winter": "true", "rain": "false"})
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.02, 0.98]))
        
    ### TODO check multiple marginals
#    def test_bucket_multiple_marginals(self):
#        resFactor = VariableElimination.bucket_marginals(self.bn, ["wet_grass", "rain"], {"winter": "true", "slippery_road": "false"})
        
        
class FactorEliminationTest(unittest.TestCase):
    
    
    def setUp(self):
        self.bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        
        
    def test_not_connected_node_without_cpt(self):
#        bn = BayesianNetwork()
        from primo2.nodes import DiscreteNode
        n = DiscreteNode("a")
        self.bn.add_node(n)
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence({"a": "False"})
        res = ft.marginals(["a"])
        #Even with evidence set, when we do not have a cpt the result should remain
        #at 0! Even if only to indicate that something might be wrong with that
        #node.
        np.testing.assert_array_almost_equal(res.get_potential(), np.array([0.0, 0.0])) 
        
    def test_empty_cpt(self):
        bn = BayesianNetwork()
        from primo2.nodes import DiscreteNode
        n1 = DiscreteNode("a")
        n2 = DiscreteNode("b")
        bn.add_node(n1)
        bn.add_node(n2)
        bn.add_edge(n1,n2)
        ft = FactorTree.create_jointree(bn)
        res = ft.marginals(["a"])
        np.testing.assert_array_almost_equal(res.get_potential(), np.array([0.0, 0.0]))
        
    def test_create_jointree(self):
        order = ["slippery_road", "wet_grass", "sprinkler", "winter", "rain"]
        ft = FactorTree.create_jointree(self.bn, order=order)
        #As above, alternatives need to be contained as well for python3
        desiredCliques = ["slippery_roadrain", "wet_grasssprinklerrain", 
                          "wet_grassrainsprinkler", "sprinklerwinterrain", 
                          "sprinklerrainwinter", "wintersprinklerrain", 
                          "winterrainsprinkler", "rainsprinklerwinter", 
                          "rainwintersprinkler"]
        self.assertEqual(len(ft.tree), 3)
        for n in ft.tree.nodes(): # was nodes_iter in networkx 1.x
            self.assertTrue(n in desiredCliques)
        
    def test_jointree_marginals(self):
        ft = FactorTree.create_jointree(self.bn)
        resFactor = ft.marginals(["winter"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.6, 0.4]))
        
    def test_jointree_marginals2(self):
        ft = FactorTree.create_jointree(self.bn)
        resFactor = ft.marginals(["slippery_road"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.364, 0.636]))
        
    def test_jointree_marginals3(self):
        ft = FactorTree.create_jointree(self.bn)
        resFactor = ft.marginals(["sprinkler"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.42, 0.58]))
        
    def test_jointree_marginals_trivial_evidence(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence({"slippery_road":"true"})
        resFactor = ft.marginals(["slippery_road"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([1.0, 0.0]))
        
    def test_jointree_evidence_trivial(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence({"wet_grass": "false"})
        resFactor = ft.marginals(["rain"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.158858, 0.841142]))
        
    def test_jointree_marginal_evidence_trivial_multiple_evidence(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence({"sprinkler": "true", "rain": "false"})
        resFactor = ft.marginals(["wet_grass"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.1, 0.9]))
    
        
    def test_jointree_marginal_evidence(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence({"winter": "true"})
        resFactor = ft.marginals(["wet_grass"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.668, 0.332]))
        
    def test_jointree_marginal_evidence_multiple_evidence(self):
        ft = FactorTree.create_jointree(self.bn)
        ft.set_evidence( {"winter": "true", "rain": "false"})
        resFactor = ft.marginals(["wet_grass"])
        np.testing.assert_array_almost_equal(resFactor.get_potential(), np.array([0.02, 0.98]))
        
    def test_jointree_marginal_soft_evidence(self):
        bn = BayesianNetwork()
        cloth = DiscreteNode("cloth", ["green","blue", "red"])
        sold = DiscreteNode("sold")
        
        bn.add_node(cloth)
        bn.add_node(sold)
        
        bn.add_edge("cloth", "sold")
        
        cloth.set_cpd(np.array([0.3,0.3,0.4]))
        sold.set_cpd(np.array([[0.4, 0.4, 0.8],
                                [0.6, 0.6, 0.2]]))
            
        tree = FactorTree.create_jointree(bn)
        tree.set_evidence({"cloth": np.array([0.7,0.25,0.05])}, softPosteriors=True)
        
        np.testing.assert_array_almost_equal(tree.marginals(["cloth"]).get_potential(), np.array([0.7,0.25,0.05]))
        np.testing.assert_array_almost_equal(tree.marginals(["sold"]).get_potential(), np.array([0.42,0.58]))
        
if __name__ == "__main__":
    #Workaround so that this script also finds the resource files when run directly
    # from within the tests folder
    import os
    os.chdir("../..")
    unittest.main()