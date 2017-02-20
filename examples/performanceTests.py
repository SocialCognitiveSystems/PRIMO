#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:20:29 2017

@author: jpoeppel
"""

import time

from primo2.io import XMLBIFParser

import primo2.inference.exact as exact
import primo2.inference.factor as factor


bn = XMLBIFParser.parse("tbals_b0_eval_study_2015-new.xbif")

f1 = factor.Factor.from_node(bn.get_node("perception"))
f2 = factor.Factor.from_node(bn.get_node("understanding"))

tree = exact.FactorTree.create_jointree(bn)


def factorTreeMarginals():
    print tree.marginals(["perception"]).potentials
    
def bucketEliminationMarginals():
    print exact.VariableElimination.bucket_marginals(bn, ["perception"]).potentials
    
    
    
t0 = time.time()
factorTreeMarginals()
print "Factor Tree took: ", time.time()-t0
                                     
t0 = time.time()
bucketEliminationMarginals()
print "Bucket took: ", time.time()-t0
    
