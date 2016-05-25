#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:16:00 2016

@author: jpoeppel
"""

import unittest

from primo.io import XMLBIFParser

from primo.inference.mcmc import MCMC
from primo.inference.mcmc import GibbsTransition
from primo.inference.mcmc import MetropolisHastingsTransition

class MCMCTest(unittest.TestCase):
    
    #TODO!
    pass
    
class GibbsTransitionTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        self.initialState = {n:n.sample_local(None) for n in self.bn.get_all_nodes()}
    
    def test_step_no_evidence(self):
        """
            It's hard to properly test this, so we will only check if the returned
            state makes sense, i.e. if all states have values that they acutally
            can take.
        """
        gibbs = GibbsTransition()
        newState = gibbs.step(self.initialState, {}, self.bn, fullChange=False)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            
    def test_step_with_evidence(self):
        """
            As above.
        """
        gibbs = GibbsTransition()
        evidence = {"wet_grass":"true"}
        self.initialState["wet_grass"] = "true"
        newState = gibbs.step(self.initialState, evidence, self.bn, fullChange=False)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            if var in evidence:
                self.assertEqual(val, evidence[var])
                
                
    def test_step_with_evidence_fullChange(self):
        """
            As above.
        """
        gibbs = GibbsTransition()
        evidence = {"sprinkler":"true"}
        self.initialState["sprinkler"] = "true"
        newState = gibbs.step(self.initialState, evidence, self.bn, fullChange=True)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            if var in evidence:
                self.assertEqual(val, evidence[var])
                
class MetropolisHastingTransitionTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = XMLBIFParser.parse("primo/tests/slippery.xbif")
        self.initialState = {n:n.sample_local(None) for n in self.bn.get_all_nodes()}
    
    def test_step_no_evidence(self):
        """
            It's hard to properly test this, so we will only check if the returned
            state makes sense, i.e. if all states have values that they acutally
            can take.
        """
        met = MetropolisHastingsTransition()
        newState = met.step(self.initialState, {}, self.bn, fullChange=False)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            
    def test_step_with_evidence(self):
        """
            As above.
        """
        met = MetropolisHastingsTransition()
        evidence = {"wet_grass":"true"}
        self.initialState["wet_grass"] = "true"
        newState = met.step(self.initialState, evidence, self.bn, fullChange=False)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            if var in evidence:
                self.assertEqual(val, evidence[var])
                
                
    def test_step_with_evidence_fullChange(self):
        """
            As above.
        """
        met = MetropolisHastingsTransition()
        evidence = {"sprinkler":"true"}
        self.initialState["sprinkler"] = "true"
        newState = met.step(self.initialState, evidence, self.bn, fullChange=True)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            if var in evidence:
                self.assertEqual(val, evidence[var])
        
        
if __name__ == "__main__":
    #Workaround so that this script also finds the resource files when run directly
    # from within the tests folder
    import os
    os.chdir("../..")
    unittest.main()