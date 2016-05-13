#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:46:05 2016

@author: jpoeppel
"""

import unittest
import numpy as np
from primo.inference.factor import Factor
from primo.nodes import DiscreteNode

class FactorTest(unittest.TestCase):


    def setUp(self):
#        cptA = np.array([0.6,0.4])
#        cptB_A = np.array([[0.9, 0.2],[0.1,0.8]])
#        cptC_B = np.array([[0.3,0.5], [0.7,0.5]])
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
        
    def test_create_from_discrete_node_error(self):
        with self.assertRaises(TypeError) as cm:
            f = Factor.from_node("Node1")
        self.assertEqual(str(cm.exception), "Only DiscreteNodes are currently supported.")
        
    def test_create_from_discrete_node(self):
        f = Factor.from_node(self.n1)
        np.testing.assert_array_equal(f.table, self.n1.cpd)
        self.assertTrue(self.n1.name in f.variables)
        for v in self.n1.values:
            self.assertTrue(v in f.values[self.n1]) #TODO find more abstract way to check values?
        
    def test_create_from_discrete_node_with_parents(self):
        f = Factor.from_node(self.n2)
        np.testing.assert_array_equal(f.table, self.n2.cpd)
        self.assertTrue(self.n2.name in f.variables)
        for p in self.n2.parentOrder:
            self.assertTrue(p in f.variables)
            for v in self.n2.parents[p].values:
                self.assertTrue(v in f.values[p])
        

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
        np.testing.assert_array_almost_equal(f3.table, res)
#        np.testing.assert_array_equal(np.shape(f3.table), np.shape(f2.table))
        # TODO: Ensure that multiplication is commutative (HARD!)
#        f4 = f2 * f1 
#        np.testing.assert_array_equal(f4.table, res)
#        np.testing.assert_array_equal(np.shape(f4.table), np.shape(f2.table))
        #Check that variables and values were correctly updated
        self.assertTrue(self.n1 in f3.variables)
#        self.assertTrue(self.n1 in f4.variables)
        self.assertTrue(self.n2 in f3.variables)
#        self.assertTrue(self.n2 in f4.variables)
        for v in self.n1.values:
            self.assertTrue(v in f3.values[self.n1])
#            self.assertTrue(v in f4.values[self.n1])
            
        for v in self.n2.values:
            self.assertTrue(v in f3.values[self.n2])
#            self.assertTrue(v in f4.values[self.n2])
        
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
        np.testing.assert_array_almost_equal(fRes.table, res)
        self.assertTrue(self.n2 in fRes.variables)
        self.assertTrue(self.n3 in fRes.variables)
        for v in self.n2.values:
            self.assertTrue(v in fRes.values[self.n2])
        for v in self.n3.values:
            self.assertTrue(v in fRes.values[self.n3])    
        
        # TODO Ensure that multiplication is commutative
#        fRes2 = f3 * f2 
#        np.testing.assert_array_equal(fRes2.table, res)
        
    def test_marginalisation(self):
        f1 = Factor.from_node(self.n1)
        f2 = Factor.from_node(self.n2)
        f3 = f1 * f2
        fRes = f3.marginalize(self.n1.name)
        res = np.array([0.34, 0.19, 0.47])
        np.testing.assert_array_almost_equal(fRes.table, res)
        self.assertFalse(self.n1.name in fRes.variables)
        self.assertFalse(self.n1.name in fRes.values)

            
    def test_marginalisation_multiple(self):
        f1 = Factor.from_node(self.n1)
        f2 = Factor.from_node(self.n2)
        f3 = f1 * f2
        fRes = f3.marginalize(["Node1", self.n2])
        res = np.array([1.0])
        np.testing.assert_array_almost_equal(fRes.table, res)

            
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
        np.testing.assert_array_almost_equal(fRes.table, res)
        self.assertFalse(self.n1 in fRes.variables)
        self.assertFalse(self.n1 in fRes.values)
        
#    def test_division(self):
#        self.fail("TODO")
    
    
if __name__ == "__main__":
    unittest.main()