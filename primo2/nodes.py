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

import random

import numpy as np


class RandomNode(object):
    """
        Base class representing a random variable in a Bayes Network.
        Only subclasses of this should actually be used in a network as this
        is underspecied for any inference algorithms.
    """
    
    def __init__(self, nodename):
        self.name = nodename
        self.cpd = 1
        
    def set_cpd(self, cpd):
        raise NotImplementedError("Called unimplemented Method")
        
        
    def __eq__(self, other):
        """
            Two random nodes are considered to be identical if they have the
            same name.
            In order for the access in dictionaries via the name to work, a
            random node is equal to its name as well.
        """
#        if isinstance(other, str):
#            return other == self.name
#        return other.name == self.name
        try:
            return other.name == self.name
        except AttributeError:
            return other == self.name
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __hash__(self):
        """
            The hash of a random node is the same as the hash of its name.
            This allows to reference nodes in dictionaries by their object
            instantiation or their name.
        """
        return hash(self.name)
        
    def __str__(self):
        return self.name
   
    def __repr__(self):
        return self.name       


class DiscreteNode(RandomNode):
    
    def __init__(self, nodename, values=("True", "False")):
        """
            Creates a new discrete node with the given name and outcomes.
            
            Paremeters
            ----------
            nodename: String
                The name which is used to reference this node
            vales: [String,], optional
                List of outcome values this discrete node has.
        """
        super(DiscreteNode,self).__init__(nodename)
        self.values = list(values) #Consider copying this if for some reason something other than strings are used in there
        self.parents = dict() #Consider removing this
        self.parentOrder = []
        self.valid = False
        self._update_dimensions()
        
    def add_parent(self, parentNode):
        """
            Adds the given node as a parent/cause node of this node. Will invalidate the
            node as its cpd will not represent the new dependence structure.
            The cpd will however be adjusted in its size.
            
            Should normally not be used directly.
            
            Parameters
            ----------
            parentNode: RandomNode
                The new parent/cause node
        """
        
        self.parents[parentNode.name] = parentNode
        self.parentOrder.append(parentNode.name)
        self._update_dimensions()
        
    def _update_dimensions(self):
        """
            Private helper function to update the dimensions of the cpd.
            Will initialice the cpd with zeros and invalidate this node.
        """
        dimensions = [len(self.values)]
        for parentName in self.parentOrder:
            dimensions.append(len(self.parents[parentName].values))
        self.cpd = np.zeros(dimensions)
        self.valid = False
        
    def remove_parent(self, parentNode):
        """
            Removes the given parent/cause node from this node.
            Although the dimensions of the cpd are adopted to fit the new 
            dependence structure, this new cpd will be initialiced to zero,
            invalidating this node.
            
            Parameters
            ---------
            parentNode: RandomNode
                The node that is to be removed as parent
        """
        del self.parents[parentNode]
        self.parentOrder.remove(parentNode.name)
        self._update_dimensions()
        
    def set_cpd(self, cpd):
        """
            Allows to set the conditional probability density(table) of this
            node directly. Will check that the dimensions of the given cpd
            is comform to the current dependency structure but will not perform
            any tests on the actual values.
            
            Parameters
            ----------
            cpd : np.array
                Table containing the conditional probabilities. Each variable 
                is represented by a dimension in the size of the number of its
                outcomes.
        """
        if np.shape(self.cpd) != np.shape(cpd):
            raise ValueError("The dimensions of the given cpd do not match the dependency structure of the node.")
        self.cpd = np.copy(cpd)
        self.valid = True
        
    def set_probability(self, valueName, prob, parentValues=None):
        """
            Function that allows to set parts of the cpt of this variable.
            IMPORTANT: Only the designated values are set, all other values
            are left untouched, meaning that the resulting cpt might be 
            invalid!! 
            
            Furthermore, if used underspecified, all corresponding entries in the
            cpt will be set to the given value: E.g. consider the binary variable
            A with binary parent B. Using A.set_probability("True", 0.4) will
            adapt the cpt to reflect 0.4 for P(A=True|B=True) AND P(A=True|B=False)
            
            Parameters
            ----------
            valueName : String
                Name of the outcome that should be set.
            prob : Float
                Probability that should be adopted.
            parentValues : Dict, optional
                A dictionary with the parent names as keys and the name of the
                corresponding value of that parent that the probability applies
                to. If underspecified, the probability will be broadcasted to
                all matching parent instantiations.
                Parent names that do not belong to this node will be ignored.
        """
        try:
            index = [self.values.index(valueName)]
        except ValueError:
            raise ValueError("This node as no value {}.".format(valueName))
        
        if not parentValues:
            parentValues = {}
        
        for parentName in self.parentOrder:
            if parentName in parentValues:
                try:
                    index.append(self.parents[parentName].values.index(parentValues[parentName]))
                except ValueError:
                    raise ValueError("Parent {} does not have values {}.".format(parentName, parentValues[parentName]))
            else:
                index.append(slice(len(self.parents[parentName].values)))
                
        self.cpd[tuple(index)] = prob
        
    def get_probability(self, value, parentValues=None):
        """
            Function to return the probability(ies) for a given value of this
            random node. If all parent values are speciefied this will return
            the single corresponding probability otherwise it will a return 
            the corresponding portion of the probability table:
            E.G. for a binary node x with a binary parent y: 
                cpt = [[0.2, 0.4],
                       [0.8, 0.6]]
                x.get_probability("True") = [0.2, 0.4]
                x.get_probability("False", {"y":["False"]}) = [0.6]
                
            Paramters
            ---------
            value: String
                The value for which the probability should be returned.
            parentValues: Dict, optional
                A dictionary with the parent names as keys and either a list
                containing the values or a single value of that parent that 
                should be included.
                Variables that are not a parent of this node will be ignored.
            
            Returns
            -------
            np.array
                A copy of the specified portion of the cpt (might be only one value).
        """
        try:
            index = [[self.values.index(value)]]
        except ValueError:
            raise ValueError("This node as no value {}.".format(value))
        
        if not parentValues:
            parentValues = {}
        
        for parentName in self.parentOrder:
            if parentName in parentValues:
                if isinstance(parentValues[parentName], list):
                    try:
                        index.append([self.parents[parentName].values.index(v) for v in parentValues[parentName]])
                    except ValueError:
                        raise ValueError("There is no conditional probability for parent {}, values {} in node {}.".format(parentName, parentValues[parentName], self.name))
                else:
                    try:
                        index.append([self.parents[parentName].values.index(parentValues[parentName])])
                    except ValueError:
                        raise ValueError("There is no conditional probability for parent {}, value {} in node {}.".format(parentName, parentValues[parentName], self.name))
            else:
                index.append(range(len(self.parents[parentName].values)))
                
        # use np.ix_ to construct the appropriate index array!
        index = np.ix_(*index)
        tmp = self.cpd[index]
        res = np.squeeze(tmp)
        return res #np.squeeze(np.copy(self.cpd[np.ix_(*index]))
        
    def _get_single_probability(self, value, parentValues=None):
        """
            Fast-path function to return the probability for a given value of this
            random node. This assumes that parentValues completely specifies 
            a single element of the cpd.
            This fast-path should mainly be used by get_markov_prob as the
            conditions for this functions are satisfied there and this saves
            quite a bit of performance when sampling!
            
            Paramters
            ---------
            value: String
                The value for which the probability should be returned.
                
            parentValues: Dict, optional
                A dictionary containing name:value pairs for all parents of this
                node. Can only be omitted if this node does not have parents.
                Variables that are not a parent of this node will be ignored.
            
            Returns
            -------
            float
                The probability of this value given the parent values.
        """
        try:
            index = [self.values.index(value)]
        except ValueError:
            raise ValueError("This node as no value {}.".format(value))
            
        if not parentValues:
            parentValues = {}
            
        for parentName in self.parentOrder:
            try:
                index.append(self.parents[parentName].values.index(parentValues[parentName]))
            except KeyError:
                raise KeyError("parentValues need to specify a value for parent {} of node: {}.".format(parentName, self.name))
            except ValueError:
                raise ValueError("There is no conditional probability for parent {}, value {} in node {}.".format(parentName, parentValues[parentName], self.name))
        return self.cpd[tuple(index)]
        
    def get_markov_prob(self, outcome, children, state, forward=False):
        """
            Computes the markov probability of the given outcome of this random variable,
            given it's markov blanket.
            
            Parameters
            ----------
            outcome: String
                The value for which the probability is to be computed.
                
            children: [RandomNode,]
                A list containing all children of this node.
                
            state: dict
                Dictionary containing the state of at least all other random nodes
                in this nodes' markov blanket as name:value pairs.
                
            forward: boolean, optional
                If forward, only the probability of this value given it's parents
                is considered, not the entire markov blanket. Useful, when generating
                an initial sample where the instantiations of the children is not
                yet known. Will ignore the given children.
                
            Returns
            -------
                float
                The probability of the given outcome given this node's markov blanket.
        """
        prob = self._get_single_probability(outcome, state)
        if not forward:
            for child in children:
                prob *= child._get_single_probability(state[child.name], state)
        return prob
        
    def sample_value(self, currentState, children, forward=False):
        """
            Returns a value drawn from the probability density given by this node
            with respect to the current state of the other nodes in the network.
            Note: Only the current Markov blanket, i.e. this variable's parents
            and children are required.
            
            Parameter
            ---------
            currentState : dict
                Dictionary containing RandomNodes as keys and their current instantiation
                as value.
                
            children : [RandomNode,]
                List of RandomNodes that have this node as parent.
                
            forward : Boolean, optional
                If forward sampling is used, the current child instantiations are
                ignored as it is assumed that we only start at the top variable
                and propagate decisions down.
                
            Returns
            -------
                String
                Name of the value this DiscreteNode has most likely adopted.
        """
        #Compute probabilities of possible outcomes
        weights = []
        for outcome in self.values:
            adaptedState = dict(currentState)
            adaptedState[self] = outcome
            weights.append(self.get_markov_prob(outcome, children, adaptedState, forward))
        
        #Perform roulette-wheel-sampling:
        rndVal = random.random()* sum(weights)
        s = 0
        for i in range(len(weights)):
            s += weights[i]
            if s >= rndVal:
                return self.values[i]
        
    def sample_local(self, currentValue):
        """
            Returns a randomly chosen possible outcome of this node. 
            
            Parameters
            ---------
            currentValue: String
                The value this node had before. This is not used for DiscreteNodes
                but might be useful in the coninuous case.
            
            Returns
            -------
                String
                The chosen value.
        """
        return random.choice(self.values)
