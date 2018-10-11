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
from primo2 import nodes
import numpy as np

class RandomNodeTest(unittest.TestCase):
    """
        This is the baseclass for all nodes. Only subclasses should be used.
    """
    def test_set_cpd(self):
        n = nodes.RandomNode("test")
        with self.assertRaises(NotImplementedError) as cm:
            n.set_cpd(None)
        self.assertEqual(str(cm.exception), "Called unimplemented method.")
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
        
    def test_set_values(self):
        n = nodes.DiscreteNode("Node1")
        n.set_values(["Value1", "Value2", "Value3"])
        self.assertEqual(n.cpd.shape, (3,))
        self.assertEqual(n.values, ["Value1", "Value2", "Value3"])
        self.assertEqual(n.valid, False)
        
    def test_set_values_with_parent(self):
        n = nodes.DiscreteNode("Node1")
        n2 = nodes.DiscreteNode("Node2")
        n.add_parent(n2)
        n.set_values(["Value1", "Value2", "Value3"])
        self.assertEqual(n.cpd.shape, (3,2))
        self.assertEqual(n.values, ["Value1", "Value2", "Value3"])
        self.assertEqual(n.valid, False)
        
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
        
    def test_set_probability(self):
        n = nodes.DiscreteNode("Node1")
        n.set_probability("True", 0.2)
        self.assertEqual(n.get_probability("True"), 0.2)
        
    def test_set_probability_with_parents(self):
        n = nodes.DiscreteNode("Node1")
        n2 = nodes.DiscreteNode("Node2")
        n2.add_parent(n)
        n2.set_probability("False", 0.4, {"Node1":"True"})
        self.assertEqual(n2.get_probability("False", {"Node1":["True"]}), 0.4)
        np.testing.assert_array_almost_equal(n2.cpd, np.array([[0.0,0.0],[0.4,0.0]]))
        
    def test_set_probability_with_underspecified_parents(self):
        n = nodes.DiscreteNode("Node1")
        n2 = nodes.DiscreteNode("Node2")
        n2.add_parent(n)
        n2.set_probability("True", 0.2)
        np.testing.assert_array_almost_equal(n2.get_probability("True"), np.array([0.2,0.2]))
        np.testing.assert_array_almost_equal(n2.cpd, np.array([[0.2,0.2],[0.0,0.0]]))
        
    def test_set_probability_with_multiple_parents(self):
        n1 = nodes.DiscreteNode("Node1")
        n2 = nodes.DiscreteNode("Node2")
        n3 = nodes.DiscreteNode("Node3")
        n3.add_parent(n1)
        n3.add_parent(n2)
        n3.set_probability("True", 0.5, {"Node1":"False", "Node2":"True"})
        
        np.testing.assert_array_almost_equal(n3.cpd, np.array([[[0.0,0.0],[0.5,0.0]],[[0.0,0.0],[0.0,0.0]]]))
        n3.set_probability("False", 0.7, {"Node2":"False"})
        np.testing.assert_array_almost_equal(n3.cpd, np.array([[[0.0,0.0],[0.5,0.0]],[[0.0,0.7],[0.0,0.7]]]))
        
    def test_set_probability_error(self):
        n = nodes.DiscreteNode("Node1")
        n2 = nodes.DiscreteNode("Node2")
        n2.add_parent(n)
        with self.assertRaises(ValueError) as cm:
            n2.set_probability("False", 0.4, {"Node1":"NotThere"})
        self.assertEqual(str(cm.exception), "Parent Node1 does not have values NotThere.")
        
    def test_set_probability_unknown_value(self):
        n = nodes.DiscreteNode("Node1")
        with self.assertRaises(ValueError) as cm:
            n.set_probability("NotThere", 0.2)
        self.assertEqual(str(cm.exception), "This node as no value NotThere.")

    def test_get_single_probabilits(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])   
        cpd = np.array([0.2,0.8])
        n.set_cpd(cpd)
        self.assertEqual(n._get_single_probability("Value1"), 0.2)
        
    def test_get_single_probability_with_parent(self):
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
        self.assertEqual(n._get_single_probability("Value1", {"Node2": "Value4"}),0.3)
        
        
        n1 = nodes.DiscreteNode("sprinkler")
        n2 = nodes.DiscreteNode("rain")
        n3 = nodes.DiscreteNode("wet_grass")
        n3.add_parent(n1)
        n3.add_parent(n2)
        cpt = np.array([[[0.95, 0.1],[0.8,0.0]], [[0.05, 0.9],[0.2, 1.0]]])
        n3.set_cpd(cpt)
        self.assertEqual(n3._get_single_probability("False", {"rain":"True", "sprinkler":"False"}),0.2)
        self.assertEqual(n3._get_single_probability("True", {"rain":"False", "sprinkler":"True"}),0.1)

        # Check other parent order!        
        n1 = nodes.DiscreteNode("sprinkler")
        n2 = nodes.DiscreteNode("rain")
        n3 = nodes.DiscreteNode("wet_grass")
        n3.add_parent(n2)
        n3.add_parent(n1)
        cpt = np.array([[[0.95, 0.8],[0.1, 0.0]],[[0.05, 0.2], [0.9, 1.0]]])
        n3.set_cpd(cpt)
        self.assertEqual(n3._get_single_probability("False", {"rain":"True", "sprinkler":"False"}),0.2)
        self.assertEqual(n3._get_single_probability("True", {"rain":"False", "sprinkler":"True"}),0.1)
     
     
    def test_get_single_probability_error(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        with self.assertRaises(ValueError) as cm:
            n._get_single_probability("Value3")
        self.assertEqual(str(cm.exception), "This node as no value {}.".format("Value3"))
        
        with self.assertRaises(ValueError) as cm:
            n._get_single_probability("Value1", {"Node2": "Value6"})
        self.assertEqual(str(cm.exception), "There is no conditional probability for parent {}, value {} in node {}.".format("Node2", "Value6", "Node1"))
        
        with self.assertRaises(KeyError) as cm:
            n._get_single_probability("Value1")
        self.assertEqual(str(cm.exception), "'parentValues need to specify a value for parent {} of node: {}.'".format("Node2", "Node1"))
        
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
        self.assertEqual(n3.get_probability("True", {"rain":["False"], "sprinkler":["True"]}),0.1)

        # Check other parent order!        
        n1 = nodes.DiscreteNode("sprinkler")
        n2 = nodes.DiscreteNode("rain")
        n3 = nodes.DiscreteNode("wet_grass")
        n3.add_parent(n2)
        n3.add_parent(n1)
        cpt = np.array([[[0.95, 0.8],[0.1, 0.0]],[[0.05, 0.2], [0.9, 1.0]]])
        n3.set_cpd(cpt)
        self.assertEqual(n3.get_probability("False", {"rain":["True"], "sprinkler":["False"]}),0.2)
        self.assertEqual(n3.get_probability("True", {"rain":["False"], "sprinkler":["True"]}),0.1)
        
    def test_get_probability_error(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        with self.assertRaises(ValueError) as cm:
            n.get_probability("Value3")
        self.assertEqual(str(cm.exception), "This node as no value {}.".format("Value3"))
        
        with self.assertRaises(ValueError) as cm:
            n.get_probability("Value1", {"Node2": ["Value6"]})
        self.assertEqual(str(cm.exception), "There is no conditional probability for parent {}, values {} in node {}.".format("Node2", "['Value6']", "Node1"))
        
        
    def test_get_probability_multiple_parents(self):
        wet = nodes.DiscreteNode("wet_grass")
        sprink = nodes.DiscreteNode("sprinkler")
        rain = nodes.DiscreteNode("rain")
        wet.add_parent(sprink)
        wet.add_parent(rain)
        cpt = np.array([[[0.95, 0.1],[0.8,0.0]],[[0.5,0.9],[0.2,1.0]]])
        wet.set_cpd(cpt)
        np.testing.assert_array_almost_equal(wet.get_probability("True", {"sprinkler":["True","False"], "rain":["True","False"]}),cpt[0,:,:])
        
    def test_get_probability_single_parent_value(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        cpd = np.array([[0.2,0.3,0.4],[0.8,0.7,0.6]])
        n.set_cpd(cpd)
        # Get only the specified probability 
        self.assertEqual(n.get_probability("Value1", {"Node2": "Value4"}),0.3)
        
    def test_get_probability_single_parent_value_error(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        cpd = np.array([[0.2,0.3,0.4],[0.8,0.7,0.6]])
        n.set_cpd(cpd)
        with self.assertRaises(ValueError) as cm:
            n.get_probability("Value1", {"Node2": "Value6"})
        self.assertEqual(str(cm.exception), "There is no conditional probability for parent {}, value {} in node {}.".format("Node2", "Value6", "Node1"))
        
        #TODO consider enforcing the same ordering of the value probabilities as speciefied
        # in the lists inside the dictionary
        
    def test_sample_value(self):
        # Testing the actual sampling is difficult because of the random element
        # involved. Therefore this test will only check if a value, that is contained
        # in the node is returned. The likelyhood of that parameter is not guaranteed!!
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        
        value = n.sample_value({}, [], forward=True)
        self.assertTrue(value in n.values)
        n.add_parent(n2)
        value = n2.sample_value({}, [n], forward=True)
        self.assertTrue(value in n2.values)
        value = n2.sample_value({"Node1":"Value2"}, [n], forward=False)
        self.assertTrue(value in n2.values)
        
    def test_sample_local(self):
        # Same problem as with sampling values. Will only check if
        # a reasonable value was drawn. Since currently uniform distribution
        # for this propositional distribution is assumed, one could generate a lot
        # of samples and check if that as approximately uniform.
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        
        value = n.sample_local("Value1")
        self.assertTrue(value in n.values)
        n.add_parent(n2)
        value = n2.sample_local("Value3")
        self.assertTrue(value in n2.values)
        value = n2.sample_local("Value5")
        self.assertTrue(value in n2.values)
        
        
    def test_get_markov_prob_forward_child(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        cpd = np.array([[0.2,0.3,0.4],[0.8,0.7,0.6]])
        n.set_cpd(cpd)
        self.assertEqual(n.get_markov_prob("Value1", [], {"Node2": "Value4"}, forward=True), 0.3)
        
    def test_get_markov_prob_forward_parent(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        cpd = np.array([[0.2,0.3,0.4],[0.8,0.7,0.6]])
        n.set_cpd(cpd)
        cpd2 = np.array([0.3, 0.5, 0.2])
        n2.set_cpd(cpd2)
        self.assertEqual(n2.get_markov_prob("Value5", [n], {"Node1": "Value2"}, forward=True), 0.2)
        
    def test_get_markov_prob(self):
        n = nodes.DiscreteNode("Node1", ["Value1", "Value2"])
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        cpd = np.array([[0.2,0.3,0.4],[0.8,0.7,0.6]])
        n.set_cpd(cpd)
        cpd2 = np.array([0.3, 0.5, 0.2])
        n2.set_cpd(cpd2)
        self.assertEqual(n2.get_markov_prob("Value5", [n], {"Node1": "Value2", "Node2": "Value5"}, forward=False), 0.12)
        
""" TODO for Continous Nodes!!!"""        

class ContinousNode(unittest.TestCase):
    pass

class UtilityNode(unittest.TestCase):
    
    def test_get_utility_error(self):
        n = nodes.UtilityNode("Node1")
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        with self.assertRaises(ValueError) as cm:
            n.get_utility({"Node2": ["Value6"]})
        self.assertEqual(str(cm.exception), "There is no utility for parent {}, value {} in node {}.".format("Node2", "['Value6']", "Node1"))
        
        
    def test_get_utility_multiple_parents(self):
        wet = nodes.UtilityNode("wet_grass")
        sprink = nodes.DiscreteNode("sprinkler")
        rain = nodes.DiscreteNode("rain")
        wet.add_parent(sprink)
        wet.add_parent(rain)
        utilities = np.array([[10,20],[100,5]])
        wet.set_utilities(utilities)
        np.testing.assert_array_almost_equal(wet.get_utility({"sprinkler":"True", "rain":"False"}),utilities[0,1])
        
    def test_get_utility_single_parent_value(self):
        n = nodes.UtilityNode("Node1")
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        utilities = np.array([5,10,35])
        n.set_utilities(utilities)
        # Get only the specified probability 
        self.assertEqual(n.get_utility({"Node2": "Value4"}),10)
        
    def test_get_utility_single_parent_value_error(self):
        n = nodes.UtilityNode("Node1")
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        utilities = np.array([5,10,35])
        n.set_utilities(utilities)
        with self.assertRaises(ValueError) as cm:
            n.get_utility({"Node2": "Value6"})
        self.assertEqual(str(cm.exception), "There is no utility for parent {}, value {} in node {}.".format("Node2", "Value6", "Node1"))
        
    def test_set_utilities_wrong_dimension(self):
        wet = nodes.UtilityNode("wet_grass")
        sprink = nodes.DiscreteNode("sprinkler")
        rain = nodes.DiscreteNode("rain")
        wet.add_parent(sprink)
        wet.add_parent(rain)
        utilities = np.array([[10,20],[100]])
        with self.assertRaises(ValueError) as cm:
            wet.set_utilities(utilities)
        self.assertEqual(str(cm.exception), "The dimensions of the given " \
                 "utility table do not match the dependency structure of the node.")
        
    def test_set_utility(self):
        n = nodes.UtilityNode("Node1")
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        n.set_utility(100, {"Node2":"Value3"})        
        self.assertEqual(n.get_utility({"Node2": "Value3"}),100)
        
        
    def test_set_utility_unknown_parentValue(self):
        n = nodes.UtilityNode("Node1")
        n2 = nodes.DiscreteNode("Node2", ["Value3", "Value4", "Value5"])
        n.add_parent(n2)
        with self.assertRaises(ValueError) as cm:
            n.set_utility(100, {"Node2":"Value6"})
        self.assertEqual(str(cm.exception), "Parent {} does not have values {}.".format("Node2", "Value6"))

"""TODO for Decision Networks!!!"""    
    
class DecisionNode(unittest.TestCase):
    
    pass

if __name__ == "__main__":
    unittest.main()