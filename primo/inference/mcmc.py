#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:18:09 2016

@author: jpoeppel
"""

import random
from primo.inference.factor import Factor

class MCMC(object):
    
    def __init__(self, bn, transitionModel=None, numSamples=1000, burnIn=1000, fullChange=False):
        """
            Creates a markov cain monte carlo instance which is used to 
            approximate marginals on a given BayesianNetwork.
            
            Parameters
            ----------
            bn : BayesianNetwork
                The network that is used to approximate the marginals
                
            transitionModel : TransitionModel, optional
                The transition model that should be used to sample the next
                state. (Default: None -> Uses Metropolis Hastings)
                
            numSamples : int, optional
                Number of samples used to approximate the probabilities. (Default: 1000)
                
            burnIn : int, optional
                The number of samples that should be discarded as burnIn. (Default: 1000)
                
            fullChange : bool, optional
                If true, a step in the sampler is considered after resampling all variables, 
                instead of only a single one (Default: False)
        """
        self.bn = bn
        self.numSamples = numSamples
        self.sampler = MarkovChainSampler(transitionModel, burnIn, fullChange)
    
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
    
    def __init__(self, transitionModel=None, burnIn=1000, fullChange = False):
        """
            Creates a markov chain sampler that creates samples given
            a network and a transition model.
            
            Parameters
            ----------
            transitionModel : TransitionModel, optional
                The model used to transition from the current state to the next state.
                If not specified, MetropolisHastingsTransition is used.
                
            burnIn : int
                Number of samples to discard before actually collecting and returning
                samples. (Default 1000)
                
            fullChange : bool, optional
                If true, a step is considered after resampling all variables, instead of
                only a single one (Default: False)
        """
        self.burnIn = burnIn
        self.fullChange = fullChange
        if not transitionModel:
#            transitionModel = GibbsTransition()
            transitionModel = MetropolisHastingsTransition()
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
        
        variablesToChange = [node for node in bn.get_all_nodes() if node not in evidence]
        curSamples = 0
        state = dict(initialState)
        while curSamples < self.burnIn:
            state = self.transitionModel.step(state, variablesToChange, bn, self.fullChange)
            curSamples += 1
        for i in xrange(numSamples):
            state = self.transitionModel.step(state, variablesToChange, bn, self.fullChange)
            yield state
    
class TransitionModel(object):
    """
        Abstract class defining a transition model.
    """
    
    def step(self, currentState, variables, bn, fullChange=False):
        raise NotImplementedError("Should be overwritten by inheriting class.")
    
class GibbsTransition(TransitionModel):
    
    def step(self, currentState, variablesToChange, bn, fullChange=False):
        """
            A method that performs "one" markov step according to Gibbs sampling.
            (See "Probabilistic Graphical Models, Daphne Koller and Nir Friedman" (p.506))
            
            Parameters
            ----------
            currentState : dict
                Dictionary containing the variables as keys and their current instantiation
                as values.
            
            variablesToChange : list
                List containing the variables for which a new state needs to be sampled.
                
            bn : BayesianNetwork
                The network that is supposed to be samples.
            
            fullChange: Boolean, optional
                If True, will create a new sample by sampling all non-evidence
                variables again. Default: False
        """
        if fullChange:
            for v in variablesToChange:
                currentState[v.name] = v.sample_value(currentState, bn.get_children(v.name))
        else:
            varToChange = random.choice(variablesToChange)
            currentState[varToChange.name] = varToChange.sample_value(currentState, bn.get_children(varToChange.name))
        
        return currentState

    
class MetropolisHastingsTransition(TransitionModel):
    
    def step(self, currentState, variablesToChange, bn, fullChange = False):
        """
            A method that performs "one" markov step according to MetropolisHasting sampling.
            (See "Probabilistic Graphical Models, Daphne Koller and Nir Friedman" (p.516))
            
            Parameters
            ----------
            currentState : dict
                Dictionary containing the variables as keys and their current instantiation
                as values.
            
            variablesToChange : list
                List containing the variables for which a new state needs to be sampled.
                
            bn : BayesianNetwork
                The network that is supposed to be samples.
            
            fullChange: Boolean, optional
                If True, will create a new sample by sampling all non-evidence
                variables again. Default: False
        """
        if fullChange:
            for v in variablesToChange:
                proposedValue = v.sample_local(currentState[v.name])
                adoptedState = dict(currentState)
                adoptedState[v.name] = proposedValue
                proposedProb = v.get_markov_prob(proposedValue, bn.get_children(v.name), adoptedState)
                currentProb = v.get_markov_prob(currentState[v.name], bn.get_children(v.name), currentState)
                
                accept = min(1.0, proposedProb/currentProb)
                if random.random() <= accept:
                    currentState[v.name] = proposedValue
        else:
            varToChange = random.choice(variablesToChange)
            proposedValue = varToChange.sample_local(currentState[varToChange])
            adoptedState = dict(currentState)
            adoptedState[varToChange.name] = proposedValue
            proposedProb = varToChange.get_markov_prob(proposedValue, bn.get_children(varToChange), adoptedState)
            currentProb = varToChange.get_markov_prob(currentState[varToChange], bn.get_children(varToChange), currentState)
            
            accept = min(1.0, proposedProb/currentProb)
            
            if random.random() <= accept:
                currentState[varToChange.name] = proposedValue
            
        return currentState
            