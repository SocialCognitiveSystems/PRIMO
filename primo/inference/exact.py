#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:52:59 2016

@author: jpoeppel
"""

from primo.inference.factor import Factor
from primo.inference.order import Orderer

class BucketElimination(object):
    
    @staticmethod
    def marginals(bn, variables, evidence={}, order=None):
        """
            Function to compute the prior or posterior marginals given evidence
            from a given Bayesian Network for the variables and their 
            values of interest. Uses the bucket elimination strategy proposed
            in "Modeling and Reasoning with Bayesian Networks" - Adnan Darwiche
            Chapter 6.10
            
            Parameters
            ----------
            bn : BayesianNetwork
                The network that is supposed to be queried.
            variables : [String,]
                List containing the names of the variables whose joint prior
                or posterior marginals are desired.
            evidence : dict, optional
                Dictionary containing the given evidence. The keys represent the
                variable names of the given evidence and the values can either
                be simple strings of the given values, or a np.array specifying
                the strength of the evidence for each outcome, e.g.:
                For the binary evidence node E
                evidence = {"E": "True"} is equivalent to {"E": np.array([1.0,0.0])}
            order : [String,], optional
                List containing the elimination order of the nodes. If order is not
                given, this algorithm computes the min degree order automatically.
                
            Returns
            -------
                Factor
                A factor containing the desired marginals
        """
        
        if not order:
            order = Orderer.get_min_degree_order(bn)
        
        for v in variables:
            order.remove(v)
            order.append(v)
            
        
        #Create buckets and fill them with factors
        buckets = []
        for v in order:
            buckets.append(Factor.from_node(bn.get_node(v)))
#            print "node {}: table: {}".format(v, bn.get_node(v).cpd)
#            print "resulting factor: ", Factor.from_node(bn.get_node(v)).table
            
        print "order: ", order
        # Add evidence to buckets
        for e in evidence.iterkeys():
            print "EVIDENCE!"
            bucketI = order.index(e)
            buckets[bucketI] = buckets[bucketI] * Factor.as_evidence(e, bn.get_node(e).values, evidence[e])
            
            
        bucketUntil = len(buckets)- len(variables)
        # Process buckets
        for i in range(bucketUntil):
            print "Cur bucket variables: ", buckets[i].variables
            print "margenilize: ", order[i]
            tmpFactor = buckets[i].marginalize(order[i])
#            print "tmpFactor.variables: ", tmpFactor.variables
            for j in range(i+1, len(order)):
                if order[j] in tmpFactor.variables:
                    buckets[j] = buckets[j] * tmpFactor
                    break
                
        for i in range(bucketUntil, len(buckets)-1):
            buckets[i+1] = buckets[i+1] * buckets[i]
            
        # Normalize evidence
        buckets[-1].normalize()

        return buckets[-1]
        
        
        