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

import lxml.etree as et
import numpy as np

from . import networks
from . import nodes

class XMLBIFParser(object):
        
    @staticmethod
    def parse(filename, ignoreProperties=True):
        bn = networks.BayesianNetwork()

        tree = et.parse(filename)
        root = tree.getroot()
        
        bn.name = root.find(".//NAME").text
        if not ignoreProperties:
            bn.meta = [prop.text for prop in root.findall("./NETWORK/PROPERTY")]
        for var in root.iter("VARIABLE"):
            curName = var.find("./NAME").text
            values = [outcome.text for outcome in var.findall("./OUTCOME")]
            curNode = nodes.DiscreteNode(curName, values) 
            
            if not ignoreProperties:
                meta = [prop.text for prop in var.findall("./PROPERTY")]
                curNode.meta = meta
            
            bn.add_node(curNode)
            
        for definition in root.iter("DEFINITION"):
            curName = definition.find("./FOR").text
            curNode = bn.get_node(curName)
            # Need to take reverse order since xmlbif files specify their
            # Table in that order. Remember to do the same inversion when writing
            # to a file!!
            for given in reversed(definition.findall("./GIVEN")):
#                curNode.add_parent(bn.get_node(given.text))
                bn.add_edge(given.text, curName)
                
            table = np.array(list(map(float, definition.find("./TABLE").text.strip().split(" "))))
            shape = [len(curNode.values)]
            for p in curNode.parentOrder:
                shape.append(len(curNode.parents[p].values))
            
            table = np.reshape(table, shape, "F")
            curNode.set_cpd(table)
            
        return bn
        
    @staticmethod
    def write(bn, filename, ignoreProperties=False):
        root = et.Element("BIF")
        root.attrib["VERSION"] = "0.3"
        network = et.SubElement(root, "NETWORK")
        netName = et.SubElement(network, "NAME")
        netName.text = bn.name
        if not ignoreProperties:
            for m in bn.meta:
                prop = et.SubElement(network, "PROPERTY")
                prop.text = m
    
        for node in bn.get_all_nodes():
            var = et.SubElement(network, "VARIABLE")
            var.attrib["TYPE"] = "nature"
      
            varName = et.SubElement(var, "NAME")
            varName.text = node.name
            for out in node.values:
                tmp = et.SubElement(var, "OUTCOME")
                tmp.text = out
            if not ignoreProperties:
                for prop in node.meta:
                    tmp = et.SubElement(var, "PROPERTY")
                    tmp.text = prop
      
            defi = et.SubElement(network, "DEFINITION")
            tmp = et.SubElement(defi, "FOR")
            tmp.text = node.name
            for parent in reversed(node.parentOrder):
                tmp = et.SubElement(defi, "GIVEN")
                tmp.text = parent
            table = et.SubElement(defi, "TABLE")
            table.text = " ".join(str(e) for e in np.reshape(node.cpd, (np.size(node.cpd)), "F"))

        with open(filename, "wb") as f:
            f.write(et.tostring(root, pretty_print=True))


class DBNSpec(object):

    @staticmethod
    def parse(filename):
        """
        Load the structure of a dynamic Bayesian network from a specification.
    
        The specification is a file in JSON-format that references the B_0 and
        B_{->} Bayesian network files in XBIF-format (via their path) and
        contains a list of the transitions connecting nodes when unrolling the
        network.

        Example:
        > {
        >     "B0": "b0_network.xbif",
        >     "TBN": "tbn_network.xbif",
        >     "transitions": [
        >         ["node_a", "node_a_p"],
        >         ["node_b", "node_b_p"]
        >     ]
        > }

        Parameters
        ----------
        filename : String
            The path of the DBN specification file.
    
        Returns
        -------
        DynamicBayesianNetwork
            The dynamic Bayesian network structure.
        """
        with open(filename) as json_data:
            spec = json.load(json_data)
        b0 = XMLBIFParser.parse(spec['B0'])
        two_tbn = XMLBIFParser.parse(spec['TBN'])
        dbn = networks.DynamicBayesianNetwork(b0, two_tbn)
        for transition in spec['transitions']:
            dbn.add_transition(transition[0], transition[1])
        return dbn
