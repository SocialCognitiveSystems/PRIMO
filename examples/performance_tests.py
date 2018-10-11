#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:20:29 2017

@author: jpoeppel
"""

import timeit

from primo2.io import XMLBIFParser

import primo2.inference.exact as exact
import primo2.inference.factor as factor


bn = XMLBIFParser.parse("dbn-example-b0.xbif")

f1 = factor.Factor.from_node(bn.get_node("perception"))
f2 = factor.Factor.from_node(bn.get_node("understanding"))

tree = exact.FactorTree.create_jointree(bn)


def factorTreeMarginals():
    tree = exact.FactorTree.create_jointree(bn)
    tree.marginals(["perception"]).potentials
    
def bucketEliminationMarginals():
    exact.VariableElimination.bucket_marginals(bn, ["perception"]).potentials
    
    
    
print("Factor Tree took: ", timeit.timeit(factorTreeMarginals, number=1000))
print("Factor Tree took: ", timeit.timeit(factorTreeMarginals, number=1000))
print("Factor Tree took: ", timeit.timeit(factorTreeMarginals, number=1000))
#                                                                                                               
print("Bucket took: ", timeit.timeit(bucketEliminationMarginals, number=1000))
print("Bucket took: ", timeit.timeit(bucketEliminationMarginals, number=1000))
print("Bucket took: ", timeit.timeit(bucketEliminationMarginals, number=1000))
    
