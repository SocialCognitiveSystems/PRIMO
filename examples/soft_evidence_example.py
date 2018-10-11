#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:52:39 2017

@author: jpoeppel
"""
import numpy as np
from primo2.networks import BayesianNetwork
from primo2.nodes import DiscreteNode

from primo2.inference.exact import VariableElimination
from primo2.inference.exact import FactorTree

from primo2.inference.mcmc import MCMC
from primo2.inference.mcmc import GibbsTransition

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

print(tree.marginals(["sold"]).get_potential())

tree.set_evidence({"cloth": np.array([0.7,0.25,0.05])})

print(tree.marginals(["sold"]).get_potential())

print(tree.marginals(["cloth"]).get_potential())

tree.set_evidence({"cloth": "green"})

print(tree.marginals(["cloth"]).get_potential())
