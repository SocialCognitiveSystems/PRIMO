#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:06:58 2016

@author: jpoeppel
"""

import numpy as np
import copy
from primo.nodes import DiscreteNode

class Factor(object):
    """
        Class representing a factor in an inference network.
        Factors can be multiplied together and variables can be summed out
        (marginalised) of factors.
    """
    
    def __init__(self):
        self.table = np.array([])
         # Use a dictionary that stores the variables as keys and their corresponding dimensions as values
        self.variables = {}
        # Use a dictionary for the value lists with the variables as keys.
        self.values = {}
        self.variableOrder = []
        
    @classmethod
    def from_node(cls, node):
        """
            Helper function that allows to create a factor from a random node.
            Currently only DiscreteNodes are supported.
            
            Paramter
            --------
            node : DiscreteNode
                The node which is used to create the factor.
                
            Returns
            -------
                Factor
                The resulting factor representing the potential at the given node.        
        """
        if not isinstance(node, DiscreteNode):
            raise TypeError("Only DiscreteNodes are currently supported.")
        res = cls()
        res.variables[node.name] = len(res.variables)
        res.variableOrder.append(node.name)
        res.values[node.name] = copy.copy(node.values)
        res.table = np.copy(node.cpd)
        for p in node.parentOrder:
            res.variables[p] = len(res.variables)
            res.variableOrder.append(p)
            res.values[p] = copy.copy(node.parents[p].values)
        return res
        
    def __mul__(self, other):
        """
            Allows multiplication of two factors. Will not modify the two
            factors in any way but return a new factor instead.
            
            Currently this is NOT commutative!!
            This means that the self-factor is used as base to create the resulting
            factor and any new variables in the other factor are added afterwards!
            They will however be added in the same order as they have been in
            the other factor.
            
            Paramter
            -------
            other : Factor
                The factor that is multiplied to this factor
                
            Returns
            -------
                Factor
                The product of this factor and the other factor.
        """
        #Create copies to avoid modification of original factors
        f1 = copy.deepcopy(self)
        f2 = copy.deepcopy(other)
        
        #Extend factor 1 by all the variables it is missing
        for v in other.variableOrder:
            if not v in f1.variables:
                ax = f1.table.ndim
                f1.variables[v] = ax
                f1.values[v] = copy.copy(other.values[v])
                f1.table = np.expand_dims(f1.table, axis = ax)
                f1.table = np.repeat(f1.table, len(other.values[v]), axis = ax)
                f1.variableOrder.append(v)
                
        
        # Ensure factor2 has the same size as factor1
        for v in f1.variableOrder:
            if not v in f2.variables:
                f2.table = np.expand_dims(f2.table, f1.variables[v])
                f2.table = np.repeat(f2.table, len(f1.values[v]), axis = f1.variables[v])
                f2.variableOrder.insert(f1.variables[v], v)
            else:
                #Roll axes around so that they align. Cannot use f2.variables[v] since
                # new axes might have been inserted in the meantime
                f2.table = np.rollaxis(f2.table, f2.variableOrder.index(v), f1.variables[v])
                
        # Pointwise multiplication which results in a factor where all instantiations
        # are compatible to the instantiations of factor1 and factor2
        # See Definition 6.3 in "Modeling and Reasoning with Bayesian Networks" - Adnan Darwiche Chapter 6
        f1.table = f1.table * f2.table

        return f1
    
    
    def marginalize(self, variables):
        """
            Function to create a new factor with the given variable summed out
            of this factor.
            
            Parameter
            ---------
            variables: String or RandomNode or [String,] or [RandomNode,]
                Either a single variable or a list of variables that are to
                be removed.
                Variables can either be addressed by their names or by the
                nodes themselves.
                
            Returns
            ------
                Factor
                A new factor where the given variables has been summed out.
        """
        
        if not hasattr(variables, '__iter__'):
            variables = [variables]
            
        res = copy.deepcopy(self)
        for v in variables:
            res.table = np.sum(res.table, axis=res.variableOrder.index(v))
            del res.values[v]
            res.variableOrder.remove(v)
            
        res.variables = {}
        #Fix variable index dictionary:
        for idx, v in enumerate(res.variableOrder):
            res.variables[v] = idx
            
        return res