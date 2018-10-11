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

"""
Created on Thu May 19 14:58:02 2016
A small collection and description of the functionality of the 
primo module
@author: jpoeppel
"""

import numpy as np
from primo2.networks import BayesianNetwork
from primo2.nodes import DiscreteNode

from primo2.inference.exact import VariableElimination
from primo2.inference.exact import FactorTree

from primo2.inference.mcmc import MCMC
from primo2.inference.mcmc import GibbsTransition

import time

#Load xmlbif 
#bn = XMLBIFParser.parse("../primo/tests/slippery.xbif")


# Or create network by hand
bn = BayesianNetwork()
wet = DiscreteNode("wet_grass", ["True","False"]) #Binary variables are assumed if no values are given
slip = DiscreteNode("slippery_road")
rain = DiscreteNode("rain")
sprinkler = DiscreteNode("sprinkler")
winter = DiscreteNode("winter")

# Add nodes to network
bn.add_node(wet)
bn.add_node(slip)
bn.add_node(rain)
bn.add_node(sprinkler)
bn.add_node(winter)

# Add edges
bn.add_edge("winter", "rain")
bn.add_edge("winter", "sprinkler")
bn.add_edge(sprinkler,"wet_grass") #Nodes and their names can usually both be used
bn.add_edge("rain", "wet_grass")
bn.add_edge("rain", "slippery_road")

# Assign cpts
# CPTS can only be set after the parental structure as been determined since
# the nodes check if the given shape of the array is correct.
# CPTS are numpy arrays with one dimension for each variable, meaning that
# the first dimension is used for the values of the variable itself and
# all the others for the parents of that variable
winter.set_cpd(np.array([0.6,0.4]))
rain.set_cpd(np.array([[0.8, 0.1],[0.2,0.9]]))
sprinkler.set_cpd(np.array([[0.2,0.75],[0.8,0.25]]))
slip.set_cpd(np.array([[0.7,0.0],[0.3,1.0]]))
wet.set_cpd(np.array([[[0.95, 0.1],[0.8,0.0]],[[0.05,0.9],[0.2,1.0]]]))


# Probabilities can also be set individually. HOWEVER, be careful when using this,
# as this will only affect the specified values and leave the others as they were!!

winter.set_probability("True", 0.6)
rain.set_probability("False", 0.2, {"winter":"True"})

# Also only partially specifying the probability is possible:

wet.set_probability("False", 0.2, {"rain":"True"})
# This will result in P(w=False|rain=True, sprinkler=True) = 0.2 AND  #
# P(w=False|rain=True, sprinkler=False) = 0.2!


# Use an inference algorithm to compute desired marginals
# Naive_marginals first computes the joint marginals over all variables
# before summing out undesired variables
res = VariableElimination.naive_marginals(bn, ["sprinkler"])
print("Marginals for sprinkler: ", res.potentials)
# The probability for a specific outcome can be queried as well
print("Probability for sprinkler=True: ", res.get_potential({"sprinkler":["True"]}))

#Posterior marginals can also be queried
res = VariableElimination.naive_marginals(bn, ["sprinkler"], {"wet_grass":"True"})
print("Marginals for sprinkler, given wet_grass=True: ", res.potentials)

#Experimental: Soft evidence (not sure how to test that yet)
# wet_grass is only true with a probability of 0.6
res = VariableElimination.naive_marginals(bn, ["sprinkler"], {"wet_grass":np.array([0.6,0.4])}) 
print("Marginals for sprinkler, given wet_grass=0.6True: ", res.potentials)



# To use the junction tree algorithm, a tree is needed first:
tree = FactorTree.create_jointree(bn)

# Set any desired evidence, this will invalidate previous messages and
# compute new ones
tree.set_evidence({"winter":"True", "slippery_road":"False"})

# Ask for desired marginals (all desired variables currently need to be present in
# a single clique within the tree, for this to work)
res = tree.marginals(["rain"])

print("P(rain=True|winter=True, slippery_road=False): ", res.get_potential({"rain":["True"]}) )


tree.set_evidence({"slippery_road":"True"})
res = tree.marginals(["slippery_road"])

print("P(slippery_road|slippery_road=True): ", res.get_potential())

# Approximate inference:
mc = MCMC(bn, transitionModel = GibbsTransition(),  numSamples=1000, burnIn=1000, fullChange=True)

start = time.time()
res = mc.marginals(["sprinkler"])
print("Approximate marginals for sprinkler: ", res.potentials)
print("Took: ", time.time()-start)

start = time.time()
res = mc.marginals(["sprinkler"], {"wet_grass":"True"})
print("Approximate marginals for sprinkler, given wet_grass=True: ", res.potentials)
print("Took: ", time.time()-start)