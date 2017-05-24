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

import json

from . import io
from . import network
from .inference import exact


class DynamicBayesianNetwork(object):
    '''
    TODO: Update docstring
    This is the implementation of a dynamic Bayesian network (also called
    temporal Bayesian network).

    Definition: DBN is a pair (B0, TwoTBN), where B0 is a BN over X(0),
    representing the initial distribution over states, and TwoTBN is a
    2-TBN for the process.
    See Koller, Friedman - "Probabilistic Graphical Models" (p. 204)

    Properties: Markov property, stationary, directed, discrete,
    acyclic (within a slice)
    '''

    def __init__(self, b0=None, two_tbn=None, transitions=None, unroll_method='SOFT_EVIDENCE'):
        super(DynamicBayesianNetwork, self).__init__()
        self._t = 0
        self._b0 = network.BayesianNetwork() if b0 is None else b0
        self._twoTBN = network.BayesianNetwork() if two_tbn is None else two_tbn
        self._transitions = []
        if transitions is not None:
            self.add_transitions(transitions)
        self.set_unroll_method(unroll_method)

    @property
    def b0(self):
        ''' Get the Bayesian network representing the initial distribution.'''
        return self._b0

    @b0.setter
    def b0(self, value):
        ''' Set the Bayesian network representing the initial distribution.'''
        self._b0 = value

    @property
    def twoTBN(self):
        return self._twoTBN

    @twoTBN.setter
    def twoTBN(self, value):
        self._twoTBN = value

    def set_unroll_method(self, name):
        if self._t == 0:
            if name == 'SOFT_EVIDENCE':
                self.unroll = self._unroll_soft_evidence
                self._ft = None
                self._transition_evidence = {}
            elif name == 'PRIOR_FEEDBACK':
                self.unroll = self._unroll_prior_feedback
            else:
                raise('Unroll method "{}" is unknown'.format(name))
        else:
            raise('Cannot switch unroll method anymore, t > 0.')

    @property
    def t(self):
        return self._t

    def add_transition(self, node, node_t):
        '''
        Mark a node as interface node.

        Keyword arguments:
        node_name -- Name of the interface node.
        node_name_t -- Name of the corresponding node in the time slice.
        '''
        node0 = self._twoTBN.get_node(node)
        node1 = self._twoTBN.get_node(node_t)
        self._transitions.append((node0, node1))

    def add_transitions(self, transitions):
        for transition in transitions:
            self.add_transition(transition[0], transition[1])

    def _unroll_prior_feedback(self, evidence=None):
        state = {}
        if self._t == 0:
            ft = exact.FactorTree.create_jointree(self._b0)
            transition_nodes = [self._b0.get_node(nt.name) for (_, nt) in self._transitions]#FIXME
        else:
            for node in ['perception_t0', 'understanding_t0', 'acceptance_t0', 'grounding_t0']: # debug
                print(self._twoTBN.get_node(node).cpd) # debug
            ft = exact.FactorTree.create_jointree(self._twoTBN)
            transition_nodes = [nt for (_, nt) in self._transitions]#FIXME
        ft.set_evidence({} if evidence is None else evidence)
        for node_t in transition_nodes:
            state[node_t] = ft.marginals([node_t]).get_potential()
            print(node_t, state[node_t]) # debug
        for (node, node_t) in self._transitions:
            self._twoTBN.get_node(node).set_cpd(state[node_t])
        self._t += 1

    def _unroll_soft_evidence(self, evidence=None):
        _evidence = {} if evidence is None else dict(evidence)
        if self._t == 0:
            self._ft = exact.FactorTree.create_jointree(self._b0)
        else:
            if self._t == 1:
                self._ft = exact.FactorTree.create_jointree(self._twoTBN)
            _evidence.update(self._transition_evidence)
        self._ft.set_evidence(_evidence, softPosteriors=True)
        self._transition_evidence.clear()
        for node, node_t in self._transitions:
            self._transition_evidence[node] = self._ft.marginals([node_t]).get_potential()
            if self._t > 0:
                print(node, self._ft.marginals([node]).get_potential()) # debug
            print(node_t, self._transition_evidence[node]) # debug
        self._t += 1


def create_DBN_from_spec(dbn_spec):
    '''
    Keyword arguments:
    dbn_spec -- is a filepath to a JSON specification of a dynamic Bayesian network

    Example:
    > {
    >     "B0": "b0_network.xbif",
    >     "TBN": "tbn_network.xbif",
    >     "transitions": [
    >         ["node_a_t0", "node_a_t"],
    >         ["node_b_t0", "node_b_t"]
    >     ]
    > }

    Returns an instantiated dynamic Bayesian network.
    '''
    with open(dbn_spec) as json_data:
        spec = json.load(json_data)

    b0 = io.XMLBIFParser.parse(spec['B0'])
    twotbn = io.XMLBIFParser.parse(spec['TBN'])
    dbn = DynamicBayesianNetwork(b0, twotbn)
    for transition in spec['transitions']:
        dbn.add_transition(transition[0], transition[1])
    return dbn
