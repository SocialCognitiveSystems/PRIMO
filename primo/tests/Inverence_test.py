#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:47:59 2016

@author: jpoeppel
"""

import unittest
from primo.network import BayesianNetwork

class VariableEliminationTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = BayesianNetwork()
        
    
    def test_marginal(self):
        self.fail("TODO")
        
    def test_marginalEvidence(self):
        self.fail("TODO")
        
        
class FactorEliminationTest(unittest.TestCase):
    
    def test_marginal(self):
        self.fail("TODO")
        
    def test_marginalEvidence(self):
        self.fail("TODO")