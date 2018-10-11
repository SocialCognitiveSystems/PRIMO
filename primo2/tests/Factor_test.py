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

from __future__ import division 
import unittest
import numpy as np
from primo2.inference.factor import Factor
from primo2.nodes import DiscreteNode, DecisionNode, UtilityNode

class FactorTest(unittest.TestCase):


    def setUp(self):
        self.n1 = DiscreteNode("Node1")
        cpt1 = np.array([0.3,0.7])
        self.n1.set_cpd(cpt1)
        self.n2 = DiscreteNode("Node2", ["ValueA","ValueB", "ValueC"])
        cpt2 = np.array([[0.2, 0.4], [0.4, 0.1], [0.4, 0.5]])
        self.n2.add_parent(self.n1)
        self.n2.set_cpd(cpt2)
        
        self.n3 = DiscreteNode("Node3", ["High", "Low"])
        cpt3 = np.array([0.8,0.2])
        self.n3.set_cpd(cpt3)
        
        self.un = UtilityNode("gains")
        
        self.un.add_parent(self.n3)
        
        self.un.set_utilities(np.array([100,10]))
        
    def test_create_from_discrete_node_error(self):
        with self.assertRaises(TypeError) as cm:
            f = Factor.from_node("Node1")
        self.assertEqual(str(cm.exception), "Only DiscreteNodes are currently supported.")
        
    def test_create_from_discrete_node(self):
        f = Factor.from_node(self.n1)
        np.testing.assert_array_equal(f.potentials, self.n1.cpd)
        self.assertTrue(self.n1.name in f)
        for v in self.n1.values:
            self.assertTrue(v in f.values[self.n1]) #TODO find more abstract way to check values?
        
    def test_create_from_discrete_node_with_parents(self):
        f = Factor.from_node(self.n2)
        np.testing.assert_array_equal(f.potentials, self.n2.cpd)
        self.assertTrue(self.n2.name in f)
        for p in self.n2.parentOrder:
            self.assertTrue(p in f)
            for v in self.n2.parents[p].values:
                self.assertTrue(v in f.values[p])
                
                
    def test_create_from_utility_node(self):
        f = Factor.from_utility_node(self.un)
        
        self.assertEqual(f.variableOrder, ["Node3"])
        np.testing.assert_array_equal(f.potentials, np.array([100,10]))
                
    def test_create_from_samples(self):
        from collections import OrderedDict
        samples = [{"A":"True", "B":"True"}, {"A":"True", "B":"False"},
                   {"A":"False", "B":"False"}, {"A":"True", "B":"True"}]
        variableValues = OrderedDict()
        variableValues["A"] = ["True","False"]
        variableValues["B"] = ["True","False"]
        res = Factor.from_samples(samples, variableValues)
        self.assertEqual(res.get_potential({"A":["True"], "B":["True"]}), 0.5)
        self.assertEqual(res.get_potential({"A":["False"], "B":["True"]}), 0)
        self.assertEqual(res.get_potential({"A":["True"], "B":["False"]}), 0.25)
        self.assertEqual(res.get_potential({"A":["False"], "B":["False"]}), 0.25)
        
    def test_as_evidence(self):
        f = Factor.as_evidence("E", ["True","False"], "True")
        np.testing.assert_array_almost_equal(f.potentials, np.array([1.0,0.0]))
        f = Factor.as_evidence("E", ["True","False"], np.array([0.8,0.2]))
        np.testing.assert_array_almost_equal(f.potentials, np.array([0.8,0.2]))
        with self.assertRaises(ValueError) as cm:
            f = Factor.as_evidence("E", ["True","False"], "NotThere")
        self.assertEqual(str(cm.exception), "Evidence NotThere is not one of the possible values (['True', 'False']) for this variable.")
        with self.assertRaises(ValueError) as cm:
            f = Factor.as_evidence("E", ["True","False"], np.array([0.1,0.2,0.3]))
        self.assertEqual(str(cm.exception), "The number of evidence strength (3) does not correspont to the number of values (2)")
                
                
    def test_invert(self):
        f1 = Factor.from_node(self.n1)
        f2 = f1.invert()
        np.testing.assert_array_almost_equal(f2.potentials, np.array([10.0/3,10.0/7]))
    
    def test_division_with_trivial_factor(self):
        f1 = Factor.get_trivial()
        f2 = Factor.from_node(self.n1)
        f3 = f2 / f1
        np.testing.assert_array_almost_equal(f3.potentials, self.n1.cpd) 
        
    def test_division_with_same_vars(self):
        f1 = Factor.from_node(self.n2)
        f2 = Factor.from_node(self.n2)
        f3 = f1 / f2
        np.testing.assert_array_almost_equal(f3.potentials, np.ones(self.n2.cpd.shape))  
        
    def test_division_with_zeros(self):
        f1 = Factor.from_node(self.n2)
        n2cpt = np.copy(self.n2.cpd)
        n2cpt[0,0] = 0
        self.n2.set_cpd(n2cpt)
        n2cpt[:,:] = 1
        n2cpt[0,0] = 0
        f2 = Factor.from_node(self.n2)
        f3 = f1 / f2
        np.testing.assert_array_almost_equal(f3.potentials, n2cpt) 
        
    def test_division_different_vars(self):
        f1 = Factor.from_node(self.n1)
        f2 = Factor.from_node(self.n2)
        with self.assertRaises(ValueError) as cm:
            f3 = f1 / f2
        self.assertEqual(str(cm.exception), "The divisor's variable are not a "\
                         "subset of the divident's variables: Divisor: {}, "\
                         "Dividend: {}".format(f2.variableOrder, f1.variableOrder))
                
    def test_multiplication_with_trivial_factor(self):
        """
            A factor may become trivial, i.e. not represent any variable anymore
            but still containing some potential.
        """
        f1 = Factor.get_trivial()
        f2 = Factor.from_node(self.n1)
        f3 = f1 * f2
        np.testing.assert_array_almost_equal(f3.potentials, self.n1.cpd)  
        f4 = f2 * f1
        np.testing.assert_array_almost_equal(f4.potentials, self.n1.cpd)  

    def test_multiplication_conditionals(self):
        """
            When multiplying a factor representing a conditional prob with
            the factor representing the variables the first one is conditioned on,
            the joint probability should be represented by the resulting factor.
            
            (This does not change the dimensions of the conditioned factor.)
            Currently it can change(i.e swap) the dimensions depending on the
            order in which the factors are multiplied.
        """
        f1 = Factor.from_node(self.n1)
        f2 = Factor.from_node(self.n2)
        res = np.array([[0.06, 0.12, 0.12], [0.28, 0.07, 0.35]])
        f3 = f1 * f2
        np.testing.assert_array_almost_equal(f3.potentials, res)
#        np.testing.assert_array_equal(np.shape(f3.potentials), np.shape(f2.potentials))
        # TODO: Ensure that multiplication is commutative (HARD!)
#        f4 = f2 * f1 
#        np.testing.assert_array_equal(f4.potentials, res)
#        np.testing.assert_array_equal(np.shape(f4.potentials), np.shape(f2.potentials))
        #Check that variables and values were correctly updated
        self.assertTrue(self.n1 in f3)
#        self.assertTrue(self.n1 in f4.variables)
        self.assertTrue(self.n2 in f3)
#        self.assertTrue(self.n2 in f4.variables)
        for v in self.n1.values:
            self.assertTrue(v in f3.values[self.n1])
#            self.assertTrue(v in f4.values[self.n1])
            
        for v in self.n2.values:
            self.assertTrue(v in f3.values[self.n2])
#            self.assertTrue(v in f4.values[self.n2])
            
    def test_division_partial_divisor(self):
        f1 = Factor()
#        f1.variables["a"] = 0
#        f1.variables["b"] = 1
        f1.variableOrder = ["a","b"]
        f1.values = {"a":["a1","a2","a3"], "b":["b1","b2"]}
        f1.potentials = np.array([[0.5,0.2],[0,0],[0.3,0.45]])
        
        f2 = Factor()
#        f2.variables["a"] = 0
        f2.variableOrder = ["a"]
        f2.values = {"a":["a1","a2","a3"]}
        f2.potentials = np.array([0.8,0,0.6])
        
        f3 = f1 / f2
        res = np.array([[0.625, 0.25],[0,0],[0.5,0.75]])
        np.testing.assert_array_almost_equal(f3.potentials, res)

    def test_multiplication_error(self):
        wet = DiscreteNode("wet_grass")
        sprink = DiscreteNode("sprinkler")
        rain = DiscreteNode("rain")
        winter = DiscreteNode("winter")
        slippery = DiscreteNode("slippery_road")
        slippery.add_parent(rain)
        wet.add_parent(sprink)
        wet.add_parent(rain)
        sprink.add_parent(winter)
        rain.add_parent(winter)

        winter.set_cpd(np.array([0.6,0.4]))
        rain.set_cpd(np.array([[0.8, 0.1],[0.2,0.9]]))
        sprink.set_cpd(np.array([[0.2,0.75],[0.8,0.25]]))
        slippery.set_cpd(np.array([[0.7,0.0],[0.3,1.0]]))
        wet.set_cpd(np.array([[[0.95, 0.1],[0.8,0.0]],[[0.05,0.9],[0.2,1.0]]]))
        
        f_sl = Factor.from_node(slippery)
        f_wg = Factor.from_node(wet)
        f_ev = Factor.as_evidence("wet_grass", ["True","False"], "False")
        f_sp = Factor.from_node(sprink)
        f_w = Factor.from_node(winter)
        f_r = Factor.from_node(rain)
        
        f_wgEm = (f_wg * f_ev).marginalize("wet_grass")
        f_test = f_sp * f_wgEm
        np.testing.assert_array_almost_equal(f_test.potentials, np.array([[[0.01, 0.18],[0.0375,0.675]], [[0.16, 0.8],[0.05,0.25]]]))
        
        
        
    def test_multiplication_unrelated_factors(self):
        """
            Multiplying unrelated factors creates a factor with a new
            dimensionality which combines all the variables from the two 
            multiplying factors.
        """
        f2 = Factor.from_node(self.n2)
        f3 = Factor.from_node(self.n3)
        fRes = f2 * f3
        res = np.array([[[0.16, 0.04], [0.32, 0.08]], [[0.32, 0.08], [0.08, 0.02]], [[0.32, 0.08], [0.4, 0.1]]])
        np.testing.assert_array_almost_equal(fRes.potentials, res)
        self.assertTrue(self.n2 in fRes.variableOrder)
        self.assertTrue(self.n3 in fRes.variableOrder)
        for v in self.n2.values:
            self.assertTrue(v in fRes.values[self.n2])
        for v in self.n3.values:
            self.assertTrue(v in fRes.values[self.n3])    
        
        # TODO Ensure that multiplication is commutative
#        fRes2 = f3 * f2 
#        np.testing.assert_array_equal(fRes2.potentials, res)
        
    def test_marginalisation(self):
        f1 = Factor.from_node(self.n1)
        f2 = Factor.from_node(self.n2)
        f3 = f1 * f2
        fRes = f3.marginalize(self.n1.name)
        res = np.array([0.34, 0.19, 0.47])
        np.testing.assert_array_almost_equal(fRes.potentials, res)
        self.assertFalse(self.n1.name in fRes)
        self.assertFalse(self.n1.name in fRes.values)

            
    def test_marginalisation_multiple(self):
        f1 = Factor.from_node(self.n1)
        f2 = Factor.from_node(self.n2)
        f3 = f1 * f2
        fRes = f3.marginalize(["Node1", self.n2])
        res = np.array([1.0])
        np.testing.assert_array_almost_equal(fRes.potentials, res)

            
    def test_marginalisation_with_node(self):
        """
            As with the nodes, for easier use one should be able to specify a
            variable both by its name as well as its random node.
        """
        f1 = Factor.from_node(self.n1)
        f2 = Factor.from_node(self.n2)
        f3 = f1 * f2
        fRes = f3.marginalize(self.n1) # This time the node itself is used
        res = np.array([0.34, 0.19, 0.47])
        np.testing.assert_array_almost_equal(fRes.potentials, res)
        self.assertFalse(self.n1 in fRes)
        self.assertFalse(self.n1 in fRes.values)
        
#    def test_division(self):
#        self.fail("TODO")
        
    def test_get_potential(self):
        """
            Factors contains "potentials" for each variable instantiation
            that it is responsible for. These potentials can be probabilities
            which might be interesting to the user.
        """
        f1 = Factor.from_node(self.n1)
        np.testing.assert_array_almost_equal(f1.get_potential(), np.array([0.3,0.7]))
        
        #TODO consider changing API of get_potential (maybe allow convencience shortcut to variable:value)
#        self.assertEqual(f1.get_potential({"Node1": ["True"]}), 0.3)
        
        f2 = Factor.from_node(self.n2)
        # Test getting a subset of the potentials. IMPORTANT: This does not
        # need to be a probability. Even if these are probabilities they might
        # be conditional probabilities (i.e. P(Node1=True|Node2=ValueB) and 
        # P(Node1=True|Node2=ValueC)) but it could also be the joint
        # pobabilities (i.e P(Node1=True, Node2=ValueB) and P(Node1=True, Node2=ValueC))
#        np.testing.assert_array_almost_equal(f2.get_potential({"Node1": ["True"], "Node2":["ValueB", "ValueC"]}), np.array([0.4,0.4]))
        
        with self.assertRaises(ValueError) as cm:
            f2.get_potential({"Node1":["False"], "Node2": ["ValueNotThere"]})
        self.assertEqual(str(cm.exception), "There is no potential for variable Node2 with values ['ValueNotThere'] in this factor.")
        
        np.testing.assert_array_almost_equal(f2.get_potential({"Node1":["False"]}), np.array([0.4,0.1,0.5]))
        
    def test_normalize(self):
        
        f1 = Factor.from_node(self.n2)
        # Disturb the potentials
        f1.potentials *= 2
        self.assertNotEqual(np.sum(f1.potentials), 1.0)
        f1.normalize()
        self.assertEqual(np.sum(f1.potentials), 1.0)
    
    
    def test_joint_factor_discrete_node(self):
        (pF, uF) = Factor.joint_factor(self.n2)
        
        np.testing.assert_array_almost_equal(pF.potentials, self.n2.cpd)
        np.testing.assert_array_almost_equal(uF.potentials, np.zeros(pF.potentials.shape))
        
    def test_joint_factor_decision_node(self):
        decisionNode = DecisionNode("dNode", decisions=["yes", "no"])
        (pF, uF) = Factor.joint_factor(decisionNode)
        
        np.testing.assert_array_almost_equal(pF.potentials, np.zeros(2))
        np.testing.assert_array_almost_equal(uF.potentials, np.zeros(pF.potentials.shape))
    
    def test_joint_factor_utility_node(self):
        utilityNode = UtilityNode("uNode")
        utilityNode.set_utilities = np.array([10,100])
        (pF, uF) = Factor.joint_factor(utilityNode)
        
        np.testing.assert_array_almost_equal(pF.potentials, np.ones(uF.potentials.shape))
        np.testing.assert_array_almost_equal(uF.potentials, utilityNode.cpd)
    
if __name__ == "__main__":
    unittest.main()