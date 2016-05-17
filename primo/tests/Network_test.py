#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:13:54 2016

@author: jpoeppel
"""

import unittest
from primo.network import BayesianNetwork
from primo.nodes import RandomNode

class BayesNetTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = BayesianNetwork()
    
    def test_add_node(self):
        n = RandomNode("Node1")
        self.assertEqual(len(self.bn), 0)
        self.bn.add_node(n)
        self.assertEqual(len(self.bn), 1)
        
    def test_add_node_only_string(self):
        with self.assertRaises(TypeError) as cm:
            self.bn.add_node("I am not a node")
        self.assertEqual(str(cm.exception), "Only subclasses of RandomNode are valid nodes.")
        
    def test_add_node_same_name(self):
        n1 = RandomNode("Node1")
        n2 = RandomNode("Node2")
        n3 = RandomNode("Node1")
        self.assertEqual(len(self.bn), 0)
        self.bn.add_node(n1)
        self.assertEqual(len(self.bn), 1)
        with self.assertRaises(ValueError) as cm:
            self.bn.add_node(n3)
        self.assertEqual(str(cm.exception), "The network already contains a node called {}".format(n1.name))
        self.assertEqual(len(self.bn), 1)
        self.bn.add_node(n2)
        self.assertEqual(len(self.bn), 2)
        
                
#    def test_addEdge(self):
#        self.fail("TODO")
                
                
    """
        TODO: ADD MISSING TESTS
    """
        
    
    
if __name__ == "__main__":
    unittest.main()