#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:24:51 2016

@author: jpoeppel
"""

import copy
from operator import itemgetter
from primo.network import BayesianNetwork

class Orderer(object):
    """
        A "static" class that provides the functionality to create elimination orders
        from Bayesian networks.
    """
    
    @staticmethod
    def get_min_degree_order(bn):
        """
            Returns the elimination order according to the min degree order algorithm
            explained in "Modeling and Reasoning with Bayesian Networks" - Adnan Darwiche
            Chapter 6.6
            
            Parameter
            ---------
            bn : BayesianNetwork
                The network for which the order is to be determined.
                
            Returns
            -------
                [String,]
                A list of variable names, representing the elimination order of 
                these variables.
        """
        if not isinstance(bn, BayesianNetwork):
            raise TypeError("Only Bayesian Networks are currently supported.")
        
        
        succs = copy.copy(bn.graph.succ)
        preds = copy.copy(bn.graph.pred)
        adjs = {n: list(succs[n])+list(preds[n]) for n in succs.keys()}
        
        
        variables = bn.get_all_node_names()
        res = []
        for i in range(len(variables)):
            degrees = {n: len(adjs) for n in adjs.keys()}
            varToElim = sorted(degrees.items(), key=itemgetter(1))[0][0]
            res.append(varToElim)
            del degrees[varToElim]
            for neigh1 in adjs[varToElim]:
                for neigh2 in adjs[varToElim]:
                    if neigh1 != neigh2:
                        if neigh1 not in adjs[neigh2]:
                            adjs[neigh1].append(neigh2)
                            ajds[neigh2].append(neigh1)
            del adjs[varToElim]
        return res

    @staticmethod
    def get_random_order(bn):
        """
            Returns the elimination order according to the order in which the nodes
            are stored internally. This is deterministic but as dictionaries
            are used to store the nodes, this might no be intuitive.
            
            Parameter
            ---------
            bn : BayesianNetwork
                The network for which the order is to be determined.
                
            Returns
            -------
                [String,]
                A list of variable names, representing the elimination order of 
                these variables.
        """                
        if not isinstance(bn, BayesianNetwork):
            raise TypeError("Only Bayesian Networks are currently supported.")
            
        return bn.get_all_node_names()
        
    