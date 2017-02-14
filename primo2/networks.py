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

import networkx as nx

from . import exceptions
from . import nodes

class BayesianNetwork(object):

    def __init__(self):
#        super(BayesianNetwork, self).__init__()
        self.graph = nx.DiGraph()
        self.node_lookup = {}
        self.name = "" #Only used to be compatible with XMLBIF

    def add_node(self, node):
        if isinstance(node, nodes.RandomNode):
            if node.name in self.node_lookup:
                raise ValueError("The network already contains a node called Node1")
            self.node_lookup[node.name]=node
            self.graph.add_node(node)
        else:
            raise TypeError("Only subclasses of RandomNode are valid nodes.")
            
    def remove_node(self, node):
        if node in self.graph:
            #Go over all children of this node
            for child in self.graph.succ[node]:
                child.remove_parent(self.node_lookup[node])
            self.graph.remove_node(node)
            del self.node_lookup[node]
    
    def remove_edge(self, fromName, toName):
        if fromName in self.graph and toName in self.graph:
            self.node_lookup[toName].remove_parent(self.node_lookup[fromName])
            self.graph.remove_edge(fromName, toName)
        

    def add_edge(self, fromName, toName):
        if fromName in self.graph and toName in self.graph:
            self.graph.add_edge(self.node_lookup[fromName], self.node_lookup[toName])
            self.node_lookup[toName].add_parent(self.node_lookup[fromName])
        else:
            raise Exception("Tried to add an Edge between two Nodes of which at least one was not contained in the Bayesnet")


    def get_node(self, node_name):
        try:
            return self.node_lookup[node_name]
        except KeyError:
            raise Exception("There is no node with name {} in the BayesianNetwork".format(node_name))

    def get_all_nodes(self):
        return self.graph.nodes()
        
    def get_all_node_names(self):
        return self.node_lookup.keys()

    def get_nodes(self, node_names=[]):
        nodes = []
        if not node_names:
            nodes = self.graph.nodes()
        else:
            for node_name in node_names:
                nodes.append(self.get_node(node_name))
        return nodes
        
    def get_children(self, nodeName):
        """
            Returns a list of all the children of the given node.
            
            Parameter
            --------
            nodeName : String or RandomNode
                The name of the node whose children are to be returned.
                
            Returns
            -------
                [RandomNode,]
                A list containing all the nodes that have the given node as parent.
        """
        return self.graph.succ[nodeName]
        
    def get_sample(self, evidence):
        sample = {}
        for e in evidence:
            sample[e] = evidence[e]
        # Make sure evidence contains RandomNodes and not only names
        for n in nx.topological_sort(self.graph):
            
            if n not in evidence:
                sample[n.name] = n.sample_value(sample, self.get_children(n), forward=True)
            
        return sample


    def clear(self):
        '''Remove all nodes and edges from the graph.
        This also removes the name, and all graph, node and edge attributes.'''
        self.graph.clear()
        self.node_lookup.clear()

    def number_of_nodes(self):
        '''Return the number of nodes in the graph.'''
        return len(self)

    def __len__(self):
        '''Return the number of nodes in the graph.'''
        return len(self.graph)
        

class DynamicBayesianNetwork(object):
    """Class representing a the structure of a dynamic Bayesian network.

    This temporal relationship is modelled as a 2-time-slice Bayesian 
    network (2-TBN; Koller & Friedman, 2009, §6.2.2). The Bayesian network
    B_0 represents the initial distribution. The Bayesian network B_{->},
    a 2-TBN, represents the process.
    """

    def __init__(self, b0=None, two_tbn=None, transitions=None):
        """Create a dynamic Bayesian network.

        Parameters
        ----------
        b0 : BayesianNetwork
            The network representing the initial distribution.
        two_tbn : BayesianNetwork
            The two-time-slice network representing the process.
        transitions : [(node, node_p),]
            A list of pairs, each of which represents one transition. 
            See add_transition for more information.
        """
        super(DynamicBayesianNetwork, self).__init__()
        self._b0 = BayesianNetwork() if b0 is None else b0
        self._two_tbn = BayesianNetwork() if two_tbn is None else two_tbn
        self._transitions = []
        if transitions is not None:
            self.add_transitions(transitions)
        
    @property
    def b0(self):
        """Get the Bayesian network B_0.
        
        Returns
        -------
        BayesianNetwork
            The network representing the initial distribution.
        """
        return self._b0

    @b0.setter
    def b0(self, bn):
        """Set the Bayesian network B_0.

        Parameters
        ----------
        bn : BayesianNetwork
            The network representing the initial distribution.
        """
        self._b0 = bn

    @property
    def two_tbn(self):
        """Get the 2-TBN B_{->}.

        Returns
        -------
            BayesianNetwork
            The two-time-slice network representing the process.
        """
        return self._two_tbn

    @two_tbn.setter
    def two_tbn(self, bn):
        """Set the 2-TBN B_{->}.

        Parameters
        ----------
        bn : BayesianNetwork
            The two-time-slice network representing the process.
        """
        self._two_tbn = bn

    def add_transition(self, node, node_p):
        """Add a transition connecting nodes when unrolling the network.

        The transition is a directed edge from node X_i (`node`) to node X_i'
        (`node_p`).

        Parameters
        ----------
        node : RandomNode, String
            The node X_i in the next time-slice.
            
        node_p: RandomNode, String
            The corresponding node X_i' in the current time-slice.

        Raises
        ------
        primo2.exceptions.StructureError
            If a node specified in the transition model cannot be found in the
            corresponding networks.
        """
        if node_p not in self._b0.get_all_nodes():
            raise exceptions.StructureError(
                'Node "{}" is not found in B_0 network "{}".'.format(
                    node_p, self._b0.name))
        if node_p not in self._two_tbn.get_all_nodes():
            raise exceptions.StructureError(
                'Node "{}" is not found in B_{->} network "{}".'.format(
                    node_p, self._two_tbn.name))
        if node not in self._two_tbn.get_all_nodes():
            raise exceptions.StructureError(
                'Node "{}" is not found in B_{{->}} network "{}".'.format(
                    node, self._two_tbn.name))
        self._transitions.append((node, node_p))

    def add_transitions(self, transitions):
        """Add multiple transitions connecting nodes when unrolling the network.

        Parameters
        ----------
        transitions : [(node, node_p),]
            A list of pairs, each of which represents one transition. 
            See add_transition for more information.
        """        
        for transition in transitions:
            self.add_transition(transition[0], transition[1])

    @property
    def transitions(self):
        """Get the transition model.

        Returns
        -------
        [(node, node_p),]
            A list of pairs, each of which represents one transition. 
            See add_transition for more information.
        """
        return self._transitions
