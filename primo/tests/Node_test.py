#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:42:51 2016

@author: jpoeppel
"""

import unittest
from primo import nodes
import numpy as np

class RandomNodeTest(unittest.TestCase):
    """
        This is the baseclass for all nodes. Only subclasses should be used.
    """
    def test_set_cpd(self):
        n = nodes.RandomNode("test")
        with self.assertRaises(NotImplementedError) as cm:
            n.set_cpd(None)
        self.assertEqual(str(cm.exception), "Called unimplemented Method")
#        
#    def test_addParent(self):
#        self.fail("TODO")
        
    def test_eq(self):
        """
            A random node is equal to itself as well as to its name object. This
            is required to allow the use of both the node as well as its name
            for lookup in dictionaries.
            For node equally only its name is used as well.
        """
        n1 = nodes.RandomNode("test")
        n2 = nodes.RandomNode("test2")
        n3 = nodes.RandomNode("test")
        self.assertEqual(n1, n3)
        self.assertNotEqual(n1,n2)
        self.assertEqual(n2, "test2")
        self.assertNotEqual(n3, "somethingElse")
        
        
    def test_hash(self):
        """
            The hash value of a random node should be the same as the hash of its name.
            This allows both the node as well as its name to be used as keys for
            dictionary lookups.
        """
        n = nodes.RandomNode("test")
        self.assertEqual(hash(n), hash("test"))
        testDict = {n: n}
        self.assertEqual(testDict[n], testDict["test"])
    
class DiscreteNode(unittest.TestCase):

    def test_create_with_values(self):
        n = nodes.DiscreteNode("Node", ["Value1", "Value"])
        self.assertEqual(n.values, ["Value1", "Value"])
        
    def test_create_with_default_values(self):
        n = nodes.DiscreteNode("Node")
        self.assertEqual(n.values, ["True", "False"])

    def test_set_cpd(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        cpd = np.array([0.2,0.8])
        n.set_cpd(cpd)
        self.assertTrue(n.valid)
        
    def test_set_cpd_invalid(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2","Value3"])
        cpd = np.array([0.2,0.8])
        with self.assertRaises(ValueError) as cm:
            n.set_cpd(cpd)
        self.assertEqual(str(cm.exception), "The dimensions of the given cpd do not match the dependency structure of the node.")
        
    def test_set_invalid_cpd_parents(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        cpd = np.array([[0.2,0.3], [0.4, 0.8], [0.7,0.6]])
        with self.assertRaises(ValueError) as cm:
            n.set_cpd(cpd)
        self.assertEqual(str(cm.exception), "The dimensions of the given cpd do not match the dependency structure of the node.")
        
        
    def test_set_cpd_two_parents(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n3 = nodes.DiscreteNode("Node3", ["Value6", "Value7"])
        n.add_parent(n2)
        n.add_parent(n3)
        cpd = np.array([[[0.2,0.4],[0.3,0.6],[0.4,0.5]],[[0.8,0.6],[0.7,0.4],[0.6,0.5]]])
        n.set_cpd(cpd)
        self.assertTrue(n.valid)
        
    def test_add_parent(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        self.assertFalse(n.valid)
        cpd = np.array([[0.2,0.3,0.4],[0.8,0.7,0.6]])
        n.set_cpd(cpd)
        self.assertTrue(n.valid)
        
    def test_remove_parent(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        cpd = np.array([[0.2,0.3,0.4],[0.8,0.7,0.6]])
        n.set_cpd(cpd)
        n.remove_parent(n2)
        self.assertFalse(n.valid)
        
    def test_get_probability(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        cpd = np.array([0.2,0.8])
        n.set_cpd(cpd)
        self.assertEqual(n.get_probability("Value1"), 0.2)
        
    def test_get_probability_with_parent(self):
        """
            Probabilities can be looked up either individually by specifying the 
            parents and their values explicitly, or as a subslice showing the
            remaining cpt for the desired value.
        """
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        cpd = np.array([[0.2,0.3,0.4],[0.8,0.7,0.6]])
        n.set_cpd(cpd)
        # Get reduced cpt for that value
        np.testing.assert_array_equal(n.get_probability("Value1"), np.array([0.2,0.3,0.4]))
        # Get only the specified probability 
        self.assertEqual(n.get_probability("Value1", {"Node2": ["Value4"]}),0.3)
        np.testing.assert_array_equal(n.get_probability("Value1", {"Node2":["Value3", "Value5"]}), np.array([0.2,0.4]))
        
        
        n1 = nodes.DiscreteNode("sprinkler")
        n2 = nodes.DiscreteNode("rain")
        n3 = nodes.DiscreteNode("wet_grass")
        n3.add_parent(n1)
        n3.add_parent(n2)
        cpt = np.array([[[0.95, 0.1],[0.8,0.0]], [[0.05, 0.9],[0.2, 1.0]]])
        n3.set_cpd(cpt)
        self.assertEqual(n3.get_probability("False", {"rain":["True"], "sprinkler":["False"]}),0.2)
        
    def test_get_probability_error(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        with self.assertRaises(ValueError) as cm:
            n.get_probability("Value3")
        self.assertEqual(str(cm.exception), "The node as no value {}".format("Value3"))
        
        with self.assertRaises(ValueError) as cm:
            n.get_probability("Value1", {"Node3": ["Value4"]})
        self.assertEqual(str(cm.exception), "There is no conditional probability for parent {}, values {}.".format("Node3", "['Value4']"))
        
#    def test_addParents(self):
#        pass
        
        
    #TODO consider enforcing the same ordering of the value probabilities as speciefied
    # in the lists inside the dictionary
        
        
""" TODO for Continous Nodes!!!"""        

class ContinousNode(unittest.TestCase):
    pass

"""TODO for Decision Networks!!!"""    
    
class DecisionNode(unittest.TestCase):
    
    pass

if __name__ == "__main__":
    unittest.main()