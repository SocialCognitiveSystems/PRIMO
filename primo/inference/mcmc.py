#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:18:09 2016

@author: jpoeppel
"""

import random
import copy
from primo.inference.factor import Factor

class MCMC(object):
    
    def __init__(self, bn, transitionModel=None, numSamples=1000, burnIn=100):
        """
            Creates a markov cain monte carlo instance which is used to 
            approximate marginals on a given BayesianNetwork.
            
            Parameters
            ----------
            bn : BayesianNetwork
                The network that is used to approximate the marginals
                
            transitionModel : TransitionModel, optional
                The transition model that should be used to sample the next
                state.
                
            numSamples : int, optional
                Number of samples used to approximate the probabilities.
                
            burnIn : int, optional
                The number of samples that should be discarded as burnIn.
        """
        self.bn = bn
        self.numSamples = numSamples
        self.sampler = MarkovChainSampler(transitionModel, burnIn)
    
    def marginals(self, variables, evidence=None):
        """
            Function that approximates the joing prior or posterior marginals for the
            given variables, potentially given some evidence.
            
            Parameters
            ---------
            variables : [String,]
                List containing the names of the desired variables.
                
            evidence : dict, optional
                Dictionary containing the evidence variables as keys and their
                instantiations as values.
                
            Returns
            -------
                Factor
                A factor over the given variables representing their joint probability 
                given the evidence.
        """
        if not evidence:
            evidence = {}
        initialState = self.bn.get_sample(evidence)
        sampleChain = self.sampler.generate_markov_chain(self.bn, self.numSamples, initialState, evidence)
        # Compute probability for variables given the samples
        variableValues = {v: self.bn.get_node(v).values for v in variables}
        res = Factor.from_samples(sampleChain, variableValues)
        
        return res
        
        
    
    
class MarkovChainSampler(object):
    
    def __init__(self, transitionModel=None, burnIn=1000):
        """
            Creates a markov chain sampler that creates samples given
            a network and a transition model.
            
            Parameters
            ----------
            transitionModel : TransitionModel, optional
                The model used to transition from the current state to the next state.
                If not specified, GibbsTransion is used.
                
            burnIn : int
                Number of samples to discard before actually collecting and returning
                samples. Default 1000
        """
        self.burnIn = burnIn
        if not transitionModel:
            transitionModel = GibbsTransition()
        self.transitionModel = transitionModel
    
    def generate_markov_chain(self, bn, numSamples, initialState, evidence=None):
        """
            Generator actually yielding the given number of samples drawn from 
            the given network starting from the initialState.
            
            Parameters
            ----------
            network : BayesianNetwork
                The network from which the samples are drawn.
                
            numSamples : int
                The number of samples this generator returns in total
                
            initialState : dict
                A dictionary containing RandomNodes as keys and their instantiation
                for the initial state of the network as values.
                
            evidence : dict, optional
                A dictionary containg the evidence variable as keys and their
                instantiation as values.
            
            Yields
            -------
                dict
                A dictionary containing the RandomNodes as keys and their current
                instantiation as values.
        """
        curSamples = 0
        state = copy.copy(initialState)
        while curSamples < self.burnIn:
            state = self.transitionModel.step(state, evidence, bn)
            curSamples += 1
        
        for i in xrange(numSamples):
            state = self.transitionModel.step(state, evidence, bn)
            yield state
    
class TransitionModel(object):
    
    def step(self, currentState, evidence, bn):
        raise NotImplementedError("Should be overwritten by inheriting class.")
    
class GibbsTransition(TransitionModel):
    
    def step(self, currentState, evidence, bn):
        #Choose a variable to change
        variables = []
        for node in bn.get_all_nodes():
            if node not in evidence:
                variables.append(node)
                
        varToChange = random.choice(variables)
        currentState[varToChange] = varToChange.sample_value(currentState, bn.get_children(varToChange))
        return currentState

    
class MetropolisHastingsTransition(TransitionModel):
    
    def step(self, **others):
        pass