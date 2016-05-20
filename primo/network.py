#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of PRIMO -- Probabilistic Inference Modules.
# Copyright (C) 2013-2015 Social Cognitive Systems Group, 
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

import primo.nodes

class BayesianNetwork(object):

    def __init__(self):
        super(BayesianNetwork, self).__init__()
        self.graph = nx.DiGraph()
        self.node_lookup = {}
        self.name = "" #Only used to be compatible with XMLBIF

    def add_node(self, node):
        if isinstance(node, primo.nodes.RandomNode):
            if node.name in self.node_lookup:
                raise ValueError("The network already contains a node called Node1")
            self.node_lookup[node.name]=node
            self.graph.add_node(node)
        else:
            raise TypeError("Only subclasses of RandomNode are valid nodes.")

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
        return self.graph.predecessors(nodeName)


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