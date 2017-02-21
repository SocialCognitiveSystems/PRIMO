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

from __future__ import division
import numpy as np
import copy
from primo2.nodes import DiscreteNode

class Factor(object):
    """
        Class representing a factor in an inference network.
        Factors can be multiplied together and variables can be summed out
        (marginalised) of factors.
    """
    def __init__(self):
        self.potentials = np.array([])
         # Use a dictionary that stores the variable names as
        # keys and their corresponding dimensions as values
        self.variables = {}
        # Use a dictionary for the value lists with the variables as keys.
        self.values = {}
        self.variableOrder = []


    def __deepcopy__(self, memodict={}):
        copy_object = Factor()
        copy_object.potentials = np.copy(self.potentials)
        copy_object.variables = dict(self.variables)
        copy_object.values = dict(self.values)
        copy_object.variableOrder = list(self.variableOrder)
        return copy_object

    def invert(self):
        res = copy.deepcopy(self)
        if len(res.variables) == 0:
            with np.errstate(divide="raise"):
                try:
                    res.potentials = 1.0/res.potentials
                except ZeroDivisionError:
                    res.potentials = 0
        else:
            with np.errstate(divide='ignore', invalid="ignore"):
                res.potentials = 1.0 / res.potentials
                res.potentials[res.potentials==np.inf] = 0
        return res

    @classmethod
    def from_samples(cls, samples, variableValues):
        """
            Creates a sample over the given variables and
            computes their potentials from the given samples. The resulting
            factor will represent the joint probability of the given variables,
            potentially given some evidence that was used when generating the
            samples.

            Parameters
            ----------
            samples : [dict,]
                List of states where each state is represented with a dictionary
                containing the variables as keys and their instantiation as value.

            variableValues: dict
                A dictionary containing the variables as keys and their value
                lists as values over which this factor is defined.

            Returns
            -------
                Factor
                A factor representing the joint probability of the given
                variables according to the given samples.
        """

        res = cls()
        shape = []
        for i, v in enumerate(variableValues):
            res.variableOrder.append(v)
            res.variables[v] = i
            res.values[v] = variableValues[v]
            shape.append(len(variableValues[v]))
        res.potentials = np.zeros(shape)
        for s in samples:
            idx = []
            for v in res.variableOrder:
                idx.append(res.values[v].index(s[v]))
            res.potentials[tuple(idx)] += 1
        res.potentials /= np.sum(res.potentials)
        return res


    @classmethod
    def get_trivial(cls, potential=1.0):
        """
            Helper function to create a trivial factor with a given potential.
            A trivial factor does not represent any variables anymore.

            Parameter
            ---------
            potential : Float, optional
                The potential this factor should be initialized with.

            Returns
            -------
                Factor
                The resulting trivial factor.
        """
        res = cls()
        res.potentials = potential
        return res

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
        res.potentials = np.copy(node.cpd)
        for p in node.parentOrder:
            res.variables[p] = len(res.variables)
            res.variableOrder.append(p)
            res.values[p] = copy.copy(node.parents[p].values)
        return res
        
    @classmethod
    def as_evidence(cls, variable, values, evidence, oldMarginals=None):
        """
            Creates an "evidence factor" which is used to introduce hart and
            soft evidence into the inference algorithms. In case of soft evidence
            the required ratio is computed in order to shift the posterior
            probability of the evidence node to the desired values.
            
            Parameters
            ---------
            variable: String
                The name of the variable that this evidence is given for.
            values: [String,]
                List of possible values/outcomes of that variable
            evidence: String or np.array
                The actual evidence that was observed. If a string is given,
                hard evidence is assumed for that given outcome. 
                Otherwise, the evidence is set according to the given array. 
                The probabilities in the array need to be in the same order as 
                given in the values.
            oldMarginals: np.array,[Float,], optional.
                This is only used when evidence is an np.array!
                List like structure containing the old marginals of the evidence
                variable. If this is given, the provided evidence will be 
                interpreted as "all things considered" soft evidence 
                (cf. "Modeling and Reasoning with Bayesian Networks" by 
                Adnan Darwiche Chapter 3.6.1), i.e. the soft evidence is 
                interpreted as the desired new posterior marginal for the 
                given variable. In this case, the likelihood ratio required to 
                reach the desired posteriors is computed and just as evidence 
                factor.
                If oldMarginals is not given, the evidence is interpreted 
                as "nothing else considered" soft evidence 
                (cf. pp.41 same book), i.e. the evidence is directly 
                interpeted as likelihood ratio.
                
            Returns
            -------
                Factor
                A factor for the given evidence variable containing potentials
                according to the strength of the evidence.
        """
        res = cls()
        res.variableOrder.append(variable)
        res.variables[variable] = 0
        res.values[variable] = copy.copy(values)
        if not isinstance(evidence, np.ndarray):
            if not evidence in values:
                raise ValueError("Evidence {} is not one of the possible " \
                                 "values ({}) for this variable."
                                .format(evidence, values))
            res.potentials = np.zeros(len(values))
            res.potentials[values.index(evidence)] = 1.0
        else:
            if len(evidence) != len(values):
                raise ValueError("The number of evidence strength ({}) " \
                            "does not correspont to the number of values ({})"
                            .format(len(evidence),len(values)))
            if oldMarginals is not None:
                #Interpret given (soft) evidence as desired posterior
                #In this case construct the required likelihood ratio
                #loosely according to Bayesian Artificial Intelligence by 
                #Kevin B. Korb and Ann E. Nicholson (pp.63), but extended to
                #work for non-binary values and hard evidence as well
                res.potentials = np.ones(np.shape(evidence))
                #Select the highest value as maximum so that this will also work
                #for specifying hard evidence the same way as soft evidence
                refIndex = np.argmax(evidence)
                refEvidence = np.max(evidence)
                for i,v in enumerate(values):
                    if i == refIndex:
                        #Just fix the reference Value to 1.
                        #Since only the ratio between the different evidence 
                        #values influences the posterior, we can choose one 
                        #value and just adjust the others relative to this.
                        res.potentials[i] = 1
                    else:
                        #Catch the case that oldValues[i] == 0
                        #=> in that case potential[i] = 0
                        if oldMarginals[i] == 0:
                            res.potentials[i] = 0
                        else:
                            res.potentials[i] = evidence[i]/oldMarginals[i] \
                                              *oldMarginals[refIndex]/refEvidence 
            else:
                #Interpet given (soft) evidence as proportion
                res.potentials = np.copy(evidence)
        return res
        
    def __truediv__(self, other):
        """
            Allows division of two factors by inverting the divisor and 
            multiplying it to the dividend. The divisor's variables must be a 
            subset of the dividend's variables for this to work since the 
            expected behaviour is not specified otherwise.
            
            Parameter
            --------
            other : Factor
                The factor that this factor is divided with.
                
            Returns
            -------
                Factor
                A new factor representing the quotient of the division.
        """
        if not set(other.variableOrder).issubset(set(self.variableOrder)):
            raise ValueError("The divisor's variable are not a subset of the " \
                             "divident's variables: Divisor: {}, Dividend: {}"
                            .format(other.variableOrder, self.variableOrder))
        return self * other.invert()
    
    #TODO consider removing this
    def __truediv__OLD(self, other):
        """
            Allows division of two factors. Currently other can only be a 
            trivial factor or a factor containing the same variables as this
            one!
            
            Parameter
            --------
            other : Factor
                The factor that this factor is divided with.
                
            Returns
            -------
                Factor
                A new factor representing the quotient of the division.
        """
        
        f1 = copy.deepcopy(self)
        f2 = copy.deepcopy(other)
        
        if len(f1.variables) == len(f2.variables) == 0:
            if f2.potentials != 0:
                f1.potentials = f1.potentials / f2.potentials
            else:
                f1.potentials = 0
            return f1
            
        if len(f2.variables) == 0:
            with np.errstate(divide='ignore'):
                f1.potentials = f1.potentials / f2.potentials
                f1.potentials[f1.potentials==np.inf] = 0
                f1.potentials[np.isnan(f1.potentials)] = 0 # Required for 0/0
            return f1
        
        if set(f1.variableOrder) != set(f2.variableOrder):
            print f1.variableOrder
            print f2.variableOrder
            raise ValueError("The divisor has different variables than the dividend.")
            
        if f1.variableOrder != f2.variableOrder:
            #Turn f2 variables to match those of f1
            for v in f1.variableOrder:
                f2.potentials = np.rollaxis(f2.potentials, f2.variableOrder.index(v), f1.variables[v])
                # Fix f2 variable order for further variables
                f2.variableOrder.remove(v)
                f2.variableOrder.insert(f1.variables[v], v)
                for idx, v in enumerate(f2.variableOrder):
                    f2.variables[v] = idx
                    
        with np.errstate(divide='ignore', invalid="ignore"):
            f1.potentials = f1.potentials / f2.potentials
            f1.potentials[f1.potentials==np.inf] = 0
            f1.potentials[np.isnan(f1.potentials)] = 0 # Required for 0/0
        return f1           
        
        
    def __mul2__(self, other):
        """
            Allows multiplication of two factors. Will not modify the two
            factors in any way but return a new factor instead.
            
            Currently this is NOT commutative!!
            This means that the self-factor is used as base to create the resulting
            factor and any new variables in the other factor are added afterwards!
            They will however be added in the same order as they have been in
            the other factor.
            
            Also value order is currently NOT checked between the two factors,
            i.e. if one factor orders the values of some variable as "True", 
            "False" and another sorts them "False", "True" 
            the results will be wrong!
            
            Paramter
            -------
            other : Factor
                The factor that is multiplied to this factor
                
            Returns
            -------
                Factor
                The product of this factor and the other factor.
        """
        
        #TODO Try performing the multiplication with views into the
        #potentials instead of copying both factors
        
        #Create copies to avoid modification of original factors
        f1 = copy.deepcopy(self)
        f2 = copy.deepcopy(other)
        
        # Shortcuts for trivial factors
        if len(f1.variables) == 0:
            f2.potentials = f1.potentials * f2.potentials
            return f2
            
        if len(f2.variables) == 0:
            f1.potentials = f1.potentials * f2.potentials
            return f1
        
        #Extend factor 1 by all the variables it is missing
        for v in other.variableOrder:
            if not v in f1.variables:
                ax = f1.potentials.ndim
                f1.variables[v] = ax
                f1.values[v] = copy.copy(other.values[v])
                f1.potentials = np.expand_dims(f1.potentials, axis = ax)
                f1.potentials = np.repeat(f1.potentials, len(other.values[v]), axis = ax)
                f1.variableOrder.append(v)
                
        # Ensure factor2 has the same size as factor1
        for v in f1.variableOrder:
            if not v in f2.variables:
                f2.potentials = np.expand_dims(f2.potentials, f1.variables[v])
                f2.potentials = np.repeat(f2.potentials, len(f1.values[v]), axis = f1.variables[v])
                #Fix variable order!
                f2.variableOrder.insert(f1.variables[v], v)
                for idx, v in enumerate(f2.variableOrder):
                    f2.variables[v] = idx
            else:
                #Roll axes around so that they align. Cannot use f2.variables[v] since
                # new axes might have been inserted in the meantime
                f2.potentials = np.rollaxis(f2.potentials, f2.variableOrder.index(v), f1.variables[v])
                
                f2.variableOrder.remove(v)
                f2.variableOrder.insert(f1.variables[v], v)
                for idx, v in enumerate(f2.variableOrder):
                    f2.variables[v] = idx

        # Pointwise multiplication which results in a factor where all instantiations
        # are compatible to the instantiations of factor1 and factor2
        # See Definition 6.3 in "Modeling and Reasoning with Bayesian Networks" - Adnan Darwiche Chapter 6
        f1.potentials = f1.potentials * f2.potentials

        return f1
    
    def __mul__(self, other):
        """
            Experimental multiplication using views instead of creating complex instances.
            Appears to make no difference for factorTrees but does improve
            BucketElimination for some weird reason.
        """
        
#        res.variableOrder = list(set(self.variableOrder+other.variableOrder))
#        res.variables = dict(self.variables)
        # Shortcuts for trivial factors
        if len(self.variables) == 0:
            res = copy.deepcopy(other)
            res.potentials = self.potentials * res.potentials
            return res
            
        if len(other.variables) == 0:
            res = copy.deepcopy(self)
            res.potentials = res.potentials * other.potentials
            return res
        
        res = Factor()
        res.variableOrder = list(self.variableOrder)
        res.values = dict(self.values)
        extra_vars = set(other.variableOrder) - set(self.variableOrder)
        #Setup res factor based on self, extended by the missing variables from other
        if extra_vars:
            slice_ = [slice(None)] * len(self.variableOrder)
            slice_.extend([np.newaxis] * len(extra_vars))
            res.potentials = self.potentials[slice_]
            
            res.variableOrder.extend(extra_vars)
            
            for var in extra_vars:
                res.values[var] = other.values[var]
                
        else:
            res.potentials = self.potentials[:]
            
        #modify other
        f2 = copy.deepcopy(other)
        extra_vars = set(res.variableOrder) - set(f2.variableOrder)
        if extra_vars:
            slice_ = [slice(None)] * len(f2.variableOrder)
            slice_.extend([np.newaxis] * len(extra_vars))
            f2.potentials = f2.potentials[slice_]
            f2.variableOrder.extend(extra_vars)
            
        
        #TODO consider using np.transpose here to get the desired order!
#        for axis in range(res.potentials.ndim):
#            exchange_index = f2.variableOrder.index(res.variableOrder[axis])
#            f2.variableOrder[axis], f2.variableOrder[exchange_index] = f2.variableOrder[exchange_index], f2.variableOrder[axis]
#            f2.potentials = f2.potentials.swapaxes(axis, exchange_index)
            
        #Using transpose instead of manually swapping axis does not seem to
        #have any measurable effect
        swaparray = [f2.variableOrder.index(var) for var in res.variableOrder]
        f2.potentials = np.transpose(f2.potentials, swaparray)
            
        res.potentials = res.potentials * f2.potentials
        
        #Fix variable index dictionary:
        res.variables = {}
        for idx, v in enumerate(res.variableOrder):
            res.variables[v] = idx
        return res
    
    def marginalize(self, variables):
        """
            Function to create a new factor with the given variable summed out
            of this factor.
            
            Parameter
            ---------
            variables: String, RandomNode, [String,], [RandomNode,], set(String,) or set(RandomNode)
                Either a single variable or a list of variables that are to
                be removed.
                Variables can either be addressed by their names or by the
                nodes themselves.
                
            Returns
            ------
                Factor
                A new factor where the given variables has been summed out.
        """
        
        if not isinstance(variables, (list,set)):
            variables = [variables]
            
        res = copy.deepcopy(self)
        for v in variables:
            res.potentials = np.sum(res.potentials, axis=res.variableOrder.index(v))
            del res.values[v]
            res.variableOrder.remove(v)
            
        #Fix variable index dictionary:
        res.variables = {}
        for idx, v in enumerate(res.variableOrder):
            res.variables[v] = idx
            
        return res
        
    def get_potential(self, variables=None):
        """
            Function that allows to query for specifiy potentials within this 
            factor. IMPORTANT: This potential is not necessary a probability.
            
            Even if probabilities are represented, these potentials could 
            be conditional probabilities or joint probabilities of different
            variables.
            
            If variables is not given, will simply return the full potential table.
            
            Parameter
            ---------
            variables: dict, optional.
                Dictionary containing the desired variable names as keys and
                a list of instantiations of interest as values.
                
            Returns
            -------
                np.array
                The currently stored potential for the given variables and their values.
        """
        
        if not variables:
            variables = {}
        
        index = []        
        for v in self.variableOrder:
            if v in variables:
                try:
                    index.append([self.values[v].index(value) for value in variables[v]])
                except ValueError:
                    raise ValueError("There is no potential for variable {} with values {} in this factor.".format(v, variables[v]))
            else:
                index.append(range(len(self.values[v])))
                    
        if len(variables) == 0:
            return self.potentials
                  
        return np.squeeze(np.copy(self.potentials[np.ix_(*index)]))
        
    def normalize(self):
        """
            Normalizes the included potential so that they add up to 1. Should
            mainly be used internally when computing posterior marginals!
        """
        potentialSum = np.sum(self.potentials)
        if potentialSum > 0:
            self.potentials /= potentialSum