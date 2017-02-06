#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:02:10 2016

@author: jpoeppel
"""

import sys
import timeit
import copy

testfile = "/homes/jpoeppel/repo/PRIMO2/primo2/tests/slippery.xbif"
burnIn = 1000
numSamples = 1000
numRuns = 100

standardPath = copy.copy(sys.path)

sys.path.append("/homes/jpoeppel/repo/PRIMO")
sys.path.append("/homes/jpoeppel/repo/PRIMO2")

from primoOld.io import XMLBIF
from primoOld.inference.mcmc import MCMC
from primoOld.inference.mcmc import MetropolisHastingsTransitionModel
from primoOld.inference.mcmc import GibbsTransitionModel
from primoOld.inference.mcmc import ConvergenceTestSimpleCounting
from primoOld.densities import ProbabilityTable
from primoOld.inference.factor import *

bn = XMLBIF.read(testfile)
slippery = bn.get_node("slippery_road")
factorTreeFactory = FactorTreeFactory()
factorTree = factorTreeFactory.create_greedy_factortree(bn)

timePrimoFactor = timeit.timeit(lambda: factorTree.calculate_marginal([slippery]), number=numRuns)


mcmc_ask=MCMC(bn, numSamples, transition_model=MetropolisHastingsTransitionModel(), 
              convergence_test=ConvergenceTestSimpleCounting(burnIn))
timePrimoMetro = timeit.timeit(lambda: mcmc_ask.calculate_PriorMarginal([slippery],ProbabilityTable), number=numRuns)

mcmc_ask=MCMC(bn, numSamples, transition_model=GibbsTransitionModel(), 
              convergence_test=ConvergenceTestSimpleCounting(burnIn))
timePrimoGibbs = timeit.timeit(lambda: mcmc_ask.calculate_PriorMarginal([slippery],ProbabilityTable), number=numRuns)

#Reload modules

from primo2.io import XMLBIFParser
from primo2.inference.mcmc import MCMC
from primo2.inference.mcmc import MetropolisHastingsTransition
from primo2.inference.mcmc import GibbsTransition
from primo2.inference.exact import FactorTree

bn = XMLBIFParser.parse(testfile)

tree = FactorTree.create_jointree(bn)
timePrimo2Factor = timeit.timeit(lambda: tree.marginals(["slippery_road"]), number=numRuns)

mcmc_ask = MCMC(bn, transitionModel=MetropolisHastingsTransition(), numSamples=numSamples, burnIn=burnIn, fullChange=True)
timePrimo2Metro = timeit.timeit(lambda: mcmc_ask.marginals(["slippery_road"]), number=numRuns)

mcmc_ask = MCMC(bn, transitionModel=GibbsTransition(), numSamples=numSamples, burnIn=burnIn, fullChange=True)
timePrimo2Gibbs = timeit.timeit(lambda: mcmc_ask.marginals(["slippery_road"]), number=numRuns)


print("Primo took for factorTree: {}".format(timePrimoFactor))
print("Primo2 took for factorTree: {}".format(timePrimo2Factor))

print("Primo took for metropolisHasting: {}".format(timePrimoMetro))
print("Primo2 took for metropolisHasting: {}".format(timePrimo2Metro))

print("Primo took for Gibbs: {}".format(timePrimoGibbs))
print("Primo2 took for Gibbs: {}".format(timePrimo2Gibbs))