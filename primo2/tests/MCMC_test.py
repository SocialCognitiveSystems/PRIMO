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

from primo2.io import XMLBIFParser

from primo2.inference.mcmc import MCMC
from primo2.inference.mcmc import GibbsTransition
from primo2.inference.mcmc import MetropolisHastingsTransition

class MCMCTest(unittest.TestCase):
    
    #TODO!
    pass
    
class GibbsTransitionTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        self.initialState = {n:n.sample_local(None) for n in self.bn.get_all_nodes()}
    
    def test_step_no_evidence(self):
        """
            It's hard to properly test this, so we will only check if the returned
            state makes sense, i.e. if all states have values that they acutally
            can take.
        """
        gibbs = GibbsTransition()
        evidence = {}
        varsToChange = [node for node in self.bn.get_all_nodes() if not node in evidence]
        newState = gibbs.step(self.initialState, varsToChange, self.bn, fullChange=False)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            
    def test_step_with_evidence(self):
        """
            As above.
        """
        gibbs = GibbsTransition()
        evidence = {"wet_grass":"true"}
        varsToChange = [node for node in self.bn.get_all_nodes() if not node in evidence]
        self.initialState["wet_grass"] = "true"
        newState = gibbs.step(self.initialState, varsToChange, self.bn, fullChange=False)
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
        varsToChange = [node for node in self.bn.get_all_nodes() if not node in evidence]
        self.initialState["sprinkler"] = "true"
        newState = gibbs.step(self.initialState, varsToChange, self.bn, fullChange=True)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            if var in evidence:
                self.assertEqual(val, evidence[var])
                
class MetropolisHastingTransitionTest(unittest.TestCase):
    
    def setUp(self):
        self.bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        self.initialState = {n:n.sample_local(None) for n in self.bn.get_all_nodes()}
    
    def test_step_no_evidence(self):
        """
            It's hard to properly test this, so we will only check if the returned
            state makes sense, i.e. if all states have values that they acutally
            can take.
        """
        met = MetropolisHastingsTransition()
        evidence = {}
        varsToChange = [node for node in self.bn.get_all_nodes() if not node in evidence]
        newState = met.step(self.initialState, varsToChange, self.bn, fullChange=False)
        for var, val in newState.items():
            self.assertTrue(val in var.values)
            
    def test_step_with_evidence(self):
        """
            As above.
        """
        met = MetropolisHastingsTransition()
        evidence = {"wet_grass":"true"}
        varsToChange = [node for node in self.bn.get_all_nodes() if not node in evidence]
        self.initialState["wet_grass"] = "true"
        newState = met.step(self.initialState, varsToChange, self.bn, fullChange=False)
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
        varsToChange = [node for node in self.bn.get_all_nodes() if not node in evidence]
        self.initialState["sprinkler"] = "true"
        newState = met.step(self.initialState, varsToChange, self.bn, fullChange=True)
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