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
import os

import lxml.etree as et
import numpy as np

from . import networks
from . import nodes

class XMLBIFParser(object):
        
    @staticmethod
    def parse(filename, ignoreProperties=True):
        """
            Static parsing method for xbif files.
            
            Parameters
            ----------
            filename: String
                Path of the xbif file to be loaded.
                
            ignoreProperties: Boolean (optional)
                If given, will ignore any PROPERTY tags in the xbif file, 
                otherwise those elements will be stored in a meta attribute
                either on the network class or the variable node, depending
                on the nesting of the PROPERTY-tag. Default: True
            
            Returns
            -------
                BayesianNetwork
                The parsed network from the xbif file.
        """
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
        """
            Static writing method for xbif files.
            
            Parameters
            ----------
            filename: String
                Path of the xbif file to be written.
                
            ignoreProperties: Boolean (optional)
                If given, any meta information in either the network or the
                variable nodes will not be written to the xbif file, 
                otherwise a PROPERTY-tag will be created for each element in
                the meta attribute, nested either directly below the network 
                (for network meta data) or the corresponding variable (for
                variable meta data). Default: True
        """
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
    def parse(filename, ignoreProperties=True):
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
            
        ignoreProperties: Boolean (optional)
            If given, will ignore any PROPERTY tags in the xbif files, 
            otherwise those elements will be stored in a meta attribute
            either on the network class or the variable node, depending
            on the nesting of the PROPERTY-tag. Default: True
    
        Returns
        -------
        DynamicBayesianNetwork
            The dynamic Bayesian network structure.
        """
        with open(filename) as json_data:
            spec = json.load(json_data)
            
        
        basepath = os.path.dirname(os.path.abspath(filename))
        if os.path.isabs(spec['B0']):
            b0 = XMLBIFParser.parse(spec['B0'], 
                                    ignoreProperties=ignoreProperties)
        else:
            b0 = XMLBIFParser.parse(basepath + os.path.sep + spec['B0'], 
                                    ignoreProperties=ignoreProperties)
        if os.path.isabs(spec['TBN']):
            two_tbn = XMLBIFParser.parse(spec['TBN'], 
                                         ignoreProperties=ignoreProperties)
        else:
            two_tbn = XMLBIFParser.parse(basepath + os.path.sep + spec['TBN'], 
                                         ignoreProperties=ignoreProperties)
        dbn = networks.DynamicBayesianNetwork(b0, two_tbn)
        for transition in spec['transitions']:
            dbn.add_transition(transition[0], transition[1])
        return dbn


    @staticmethod
    def write(dbn, path, name, ignoreProperties=True):
        """
        Write the structure of a dynamic Bayesian network to a specification.
        This function will create a total of 3 files:
        
        1. A ".conf" specification file, which specifies which B0 and TBN
            to use and what the transitions between time steps should be.
        2. A "-bo.xbif" normal XBIF network description of the B0 network.
        3. A "-2tbn.xbif" normal XBFI network description of the TBN network.
        
        Parameters
        ----------
        dbn : DynamicBayesianNetwork
            The dynamic Network to be written
        
        path : String
            The path to the folder where to store the three files
            
        name : String
            The base name of the dynamic network, the created files will append
            their repsective suffix to this name
            
        ignoreProperties : Boolean (optional)
            If given, any meta information in either the networks or the
            variable nodes will not be written to the xbif files, 
            otherwise a PROPERTY-tag will be created for each element in
            the meta attribute, nested either directly below the network 
            (for network meta data) or the corresponding variable (for
            variable meta data). Default: True
            
        """
        b0 = dbn.b0
        tbn = dbn.two_tbn
        file_type_conf = ".conf"
        file_type_b0 = "-b0.xbif"
        file_type_2tbn = "-2tbn.xbif"

        json_data = {
            "B0": name + file_type_b0,
            "TBN": name + file_type_2tbn,
            "transitions": dbn.transitions
        }

        file_name_b0 = path + name + file_type_b0
        XMLBIFParser.write(b0, "".join(file_name_b0), 
                           ignoreProperties=ignoreProperties)

        file_name_2tbn = path + name + file_type_2tbn
        XMLBIFParser.write(tbn, "".join(file_name_2tbn), 
                           ignoreProperties=ignoreProperties)

        file_name_conf = path + name + file_type_conf
        with open(file_name_conf, "w") as f:
            json.dump(json_data, f, sort_keys=True, indent=4, 
                      ensure_ascii=ignoreProperties)
