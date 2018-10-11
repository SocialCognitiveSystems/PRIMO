#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:45:51 2017

@author: jpoeppel
"""

from __future__ import division 

import functools

from ..nodes import UtilityNode
from .factor import Factor

class VariableElimination(object):
    
    def __init__(self, decisionNetwork):
        self.net = decisionNetwork
        
    def get_decision(self, decisionNode, otherDecisions=None):
        """
            
        """
        pass

    
    def _combine_factors(self, factors):
        """
            Helper function to combine multiple joint factors.
            
            Parameters
            ----------
            factors: iterable(tuple)
                An iterable of joint factors
                
            Returns
            -------
                tuple
                A joint factor representing the combination of all the given
                factors.
        """
        def _combine_two_factors(f1, f2):
            return (f1[0]*f2[0], f1[1]+f2[1])
            
        return functools.reduce(_combine_two_factors, factors)
    
    def _marginalize_joint_factor(self, factor, variable):
        """
            Helper function to marginalize a variable from a joint factor, 
            following eq 23.6 in Koller, Friedman: Probabilistic Graphical Models.
            
            Parameters
            ---------
            factor: tuple
                The joint factor to be reduced
                
            variable: string
                The name of the variable to be marginalized out.
                
            Returns
            -------
                tuple
                A joint factor with the given variable marginalized out.
        """
        tmp = factor[0].marginalize(variable)
        return (tmp, (factor[0]*factor[1]).marginalize(variable)/tmp)

    def generalized_VE(self, joint_factors, elimination_variables):
        """
            Generalized variable elimination for joint factors in influence
            diagrams (cf. Koller, Friedman: Probabilistic Graphical Models,
            Algorithm 23.2)
            
            Parameters
            ---------
            joint_factors: set(tuple)
                A set of tuple containing probabilistic factors and utility factors
                
            elimination_variables: list
                A list of variables that should be marginalized out
                
            Returns
            -------
                tuple
                A joint factor containing a probabilistic factor and a 
                utility factor over all the variables from the initial
                joint_factors which have not been eliminated.
        """
        
        working_factors = set(joint_factors)
        for var in elimination_variables:
            relevant_factors = set([f for f in working_factors if var in f[0].variableOrder])
            tmp_factor = self._combine_factors(relevant_factors)
            marginalized_factor = self._marginalize_joint_factor(tmp_factor, var)
            
            working_factors = (working_factors - relevant_factors)
            working_factors.add(marginalized_factor)
            
        res_factor = self._combine_factors(working_factors)
        
        return res_factor
    
    def expected_utility(self, decisions=None):
        """
            Computes the expected utility fo the decision network, given the 
            provided decisions. If no decisions are given, the algorithm
            assumes that the decisionNodes' state has already been set.
            
            Parameters
            ----------
            decisions: dict (optional)
                A dictionary containing the decision variable names as keys
                and their set decision as value.
                
            Returns
            -------
                float
                The expected utility for those decisions.
        """
        if decisions is None:
            decisions = {}
        #Set given decisions
        for decision in decisions:
            decisionNode = self.net.node_lookup[decision]
#            decisionNode.clear()
            decisionNode.set_decision(decisions[decision])
            
        #Create joint factors
        factors = set([])
        for node in self.net.node_lookup.values():
            factors.add(Factor.joint_factor(node))
        
        eliminations = [node for node in self.net.node_lookup.values() if not isinstance(node, UtilityNode)]
        return self.generalized_VE(factors, eliminations)[1].potentials
        
    
    def _optimize_locally(self, decisionNodeName):
        """
            Helper function to choose the optimal decision for the given
            decisionNode. 
            
            Parameter
            ---------
            decisionNodeName: string
                The name of the decision to be optimized
                
            Returns
            -------
                string
                The optimal decision for this node
        """
        nodes = [node for node in self.net.node_lookup.keys() 
                            if node != decisionNodeName and 
                            not isinstance(self.net.node_lookup[node], UtilityNode) and 
                            node not in self.net.node_lookup[decisionNodeName].parents]
        decisionNode = self.net.node_lookup[decisionNodeName]
        #Create joint factors
        factors = set([])
        for node in self.net.node_lookup.values():
            factors.add(Factor.joint_factor(node))
            
        reducedFactor = self.generalized_VE(factors, nodes)
        
        
        argmax = None
        maxVal = None
        for decision in decisionNode.values:
            eu = reducedFactor[1].get_potential({decisionNodeName: [decision]})
            if maxVal is None or eu > maxVal:
                maxVal = eu
                argmax = decision
        
        return argmax
        
    
    def get_optimal_decisions(self, decisionOrder, fixedDecisions=None):
        """
            Iteratively computes the optimal decisions for all given decision
            nodes according to the "Iterated optimization for influence diagrams
            with acyclic relevance graphs" 
            (cf. Koller, Friedman: Probabilistic Graphical Models, alg 23.3)
            
            Parameters
            ----------
            decisionOrder: list
                A list of decisionNode names giving the ordering according to
                the relevance graph of the network. (also known as the
                inverted partial ordering of the decisions)
                
            fixedDecisions: dict (optional)
                A dictionary which allows to optionally specify certain
                decisions which should not be optimized
                
            Returns
            -------
                dict
                A dictionary containing the optimal decision for each of the
                decisionNodes
        """
        if fixedDecisions is None:
            fixedDecisions = {}
            
        #Initialize random fully mixed strategy
        for decisionNode in decisionOrder:
            if not decisionNode in fixedDecisions:
                self.net.node_lookup[decisionNode].fully_mixed()
            else:
                self.net.node_lookup[decisionNode].set_decision(fixedDecisions[decisionNode])
        
        solution = {}
        for decisionNode in decisionOrder:
            localDecision = self._optimize_locally(decisionNode)
            solution[decisionNode] = localDecision
            
        return solution
            
        
        