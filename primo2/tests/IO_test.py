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

import os
import unittest
import numpy as np

from primo2.networks import BayesianNetwork
from primo2.io import XMLBIFParser, DBNSpec
from primo2.nodes import DiscreteNode

class XMLBIFTest(unittest.TestCase):
    

    def test_readXMLBIF(self):
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        
        nodes = bn.get_all_nodes()        
        self.assertTrue("slippery_road" in nodes)
        self.assertTrue("sprinkler" in nodes)
        self.assertTrue("rain" in nodes)
        self.assertTrue("wet_grass" in nodes)
        self.assertTrue("winter" in nodes)
        self.assertEqual(len(nodes), 5)
        slipperyNode = bn.get_node("slippery_road")
        self.assertTrue("rain" in slipperyNode.parents)
        sprinklerNode = bn.get_node("sprinkler")
        self.assertTrue("winter" in sprinklerNode.parents)
        rainNode = bn.get_node("rain")
        self.assertTrue("winter" in rainNode.parents)
        cpt = np.array([[0.8,0.1],[0.2,0.9]])        
        np.testing.assert_array_almost_equal(rainNode.cpd, cpt)
        
        wetNode = bn.get_node("wet_grass")
        self.assertTrue("sprinkler" in wetNode.parents)
        self.assertTrue("rain" in wetNode.parents)
        self.assertTrue("true" in wetNode.values)
        cpt = np.array([[[0.95, 0.8],[0.1,0.0]], [[0.05, 0.2],[0.9, 1.0]]])
        self.assertEqual(wetNode.get_probability("false", {"rain":["true"], "sprinkler":["false"]}),0.2)
        self.assertEqual(wetNode.get_probability("true", {"rain":["false"], "sprinkler":["true"]}),0.1)
        
#        np.testing.assert_array_almost_equal(wetNode.cpd, cpt)
        
    def test_readXMLBIF_different_parent_sizes(self):
        bn = XMLBIFParser.parse("primo2/tests/testfile.xbif")
#        nodes = bn.get_all_nodes()
        johnNode = bn.get_node("John_calls")
        cpt = np.array([[[0.8,0.5,0.7],[0.6,0.2,0.1]],[[0.2,0.5,0.3],[0.4,0.8,0.9]]])
        np.testing.assert_array_almost_equal(johnNode.cpd, cpt)
        
    def test_readXMLBIF_with_variable_properties(self):
        bn = XMLBIFParser.parse("primo2/tests/testfile.xbif", ignoreProperties=False)
        
        johnNode = bn.get_node("John_calls")
        self.assertEqual(len(johnNode.meta), 1)
        self.assertTrue("position" in johnNode.meta[0])
        
        alarmNode = bn.get_node("Alarm")
        
        self.assertEqual(len(alarmNode.meta), 2)
        self.assertTrue("position" in alarmNode.meta[0])
        self.assertEqual("Random meta test", alarmNode.meta[1])
        
    def test_readXMLBIF_with_variable_properties_ignored(self):
        bn = XMLBIFParser.parse("primo2/tests/testfile.xbif", ignoreProperties=True)
        johnNode = bn.get_node("John_calls")
        self.assertEqual(len(johnNode.meta), 0)
        
        alarmNode = bn.get_node("Alarm")
        self.assertEqual(len(alarmNode.meta), 0)
        
    def test_readXMLBIF_with_network_properties(self):
        bn = XMLBIFParser.parse("primo2/tests/testfile.xbif", ignoreProperties=False)
        
        self.assertEqual(len(bn.meta), 2)
        self.assertEqual("Random network property", bn.meta[0])
        self.assertEqual("Author jpoeppel", bn.meta[1])
        
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif", ignoreProperties=False)
        self.assertEqual(len(bn.meta), 0)
        
    def test_readXMLBIF_with_network_properties_ignored(self):
        bn = XMLBIFParser.parse("primo2/tests/testfile.xbif", ignoreProperties=True)
        self.assertEqual(len(bn.meta), 0)
        
        
        
        
    def test_writeXMLBIF_simple(self):
        path= "primo2/tests/test.xbif"
        bn = BayesianNetwork()
        n1 = DiscreteNode("a")
        n2 = DiscreteNode("b", ["happy", "sad"])
        bn.add_node(n1)
        bn.add_node(n2)
        bn.add_edge(n1,n2)
        XMLBIFParser.write(bn, path)

        bn2 = XMLBIFParser.parse(path)
        for n in bn2.get_all_nodes():
            tmpn = bn.get_node(n)
            for value in tmpn.values:
                self.assertTrue(value in n.values)
            for p in tmpn.parents.keys():
                self.assertTrue(p in n.parents)
            np.testing.assert_almost_equal(tmpn.cpd, n.cpd)
        # remove testfile
        import os
        os.remove(path)       
            
    def test_writeXMLBIF(self):
        testPath = "primo2/tests/testSlippery.xbif"
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        XMLBIFParser.write(bn, testPath)
        bn2 = XMLBIFParser.parse(testPath)
        for n in bn2.get_all_nodes():
            tmpn = bn.get_node(n)
            for value in tmpn.values:
                self.assertTrue(value in n.values)
            for i,p in enumerate(tmpn.parentOrder):
                self.assertEqual(p, n.parentOrder[i])
            np.testing.assert_almost_equal(tmpn.cpd, n.cpd)
        # remove testfile
        import os
        os.remove(testPath)
        
    def test_writeXMLBIF_with_network_properties_ignored(self):
        testPath = "primo2/tests/testSlippery.xbif"
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        bn.meta = ["Dummy property"]
        XMLBIFParser.write(bn, testPath, ignoreProperties=True)
        bn2 = XMLBIFParser.parse(testPath, ignoreProperties=False)
        self.assertEqual(len(bn2.meta),0)
        self.assertEqual("Dummy property", bn.meta[0])
        os.remove(testPath)
        
    def test_writeXMLBIF_with_network_properties(self):
        testPath = "primo2/tests/testSlippery.xbif"
        bn = XMLBIFParser.parse("primo2/tests/slippery.xbif")
        bn.meta = ["Dummy property"]
        XMLBIFParser.write(bn, testPath, ignoreProperties=False)
        bn2 = XMLBIFParser.parse(testPath, ignoreProperties=False)
        self.assertEqual(len(bn2.meta),1)
        self.assertEqual("Dummy property", bn.meta[0])
        os.remove(testPath)
        
    def test_writeXMLBIF_with_variable_properties_ignored(self):
        testPath = "primo2/tests/test_testfile.xbif"
        readPath = "primo2/tests/testfile.xbif"
        bn = XMLBIFParser.parse(readPath, ignoreProperties=False)
        XMLBIFParser.write(bn, testPath, ignoreProperties=True)
        bn2 = XMLBIFParser.parse(testPath, ignoreProperties=False)
        johnNode = bn2.get_node("John_calls")
        self.assertEqual(len(johnNode.meta), 0)
        
        alarmNode = bn2.get_node("Alarm")
        self.assertEqual(len(alarmNode.meta), 0)
        os.remove(testPath)
    
    def test_writeXMLBIF_with_variable_properties(self):
        testPath = "primo2/tests/test_testfile.xbif"
        bn = XMLBIFParser.parse("primo2/tests/testfile.xbif", ignoreProperties=False)
        XMLBIFParser.write(bn, testPath, ignoreProperties=False)
        bn2 = XMLBIFParser.parse(testPath, ignoreProperties=False)
        johnNode = bn2.get_node("John_calls")
        self.assertEqual(len(johnNode.meta), 1)
        self.assertTrue("position" in johnNode.meta[0])
        
        alarmNode = bn2.get_node("Alarm")
        
        self.assertEqual(len(alarmNode.meta), 2)
        self.assertTrue("position" in alarmNode.meta[0])
        self.assertEqual("Random meta test", alarmNode.meta[1])
        os.remove(testPath)
        
class DBNSpecTest(unittest.TestCase):
    
    def test_parseDBN(self):
        dbn = DBNSpec.parse("primo2/tests/dbn-test.conf")
        self.assertEqual(dbn._b0.name, "Test_DBN_B0")
        self.assertEqual(dbn._two_tbn.name, "Test_DBN_2TBN")
    
    def test_parseDBN_local_dir(self):
        os.chdir("primo2/tests")
        dbn = DBNSpec.parse("dbn-test.conf")
        self.assertEqual(dbn._b0.name, "Test_DBN_B0")
        self.assertEqual(dbn._two_tbn.name, "Test_DBN_2TBN")
        os.chdir("../..")
        
    def test_parseDBN_relative(self):
        import shutil
        shutil.copyfile("primo2/tests/dbn-test-b0.xbif", "primo2/dbn-test-b0.xbif")
        shutil.copyfile("primo2/tests/dbn-test-2tbn.xbif", "primo2/dbn-test-2tbn.xbif")
        dbn = DBNSpec.parse("primo2/tests/dbn-test-relative.conf")
        self.assertEqual(dbn._b0.name, "Test_DBN_B0")
        self.assertEqual(dbn._two_tbn.name, "Test_DBN_2TBN")
        #Clean up
        os.remove("primo2/dbn-test-b0.xbif")
        os.remove("primo2/dbn-test-2tbn.xbif")
        
    def test_parseDBN_absolute_path(self):
        if os.path.exists("/tmp"):
            import shutil
            shutil.copyfile("primo2/tests/dbn-test-b0.xbif", "/tmp/dbn-test-b0.xbif")
            shutil.copyfile("primo2/tests/dbn-test-2tbn.xbif", "/tmp/dbn-test-2tbn.xbif")
            dbn = DBNSpec.parse("primo2/tests/dbn-test-abs.conf")
            self.assertEqual(dbn._b0.name, "Test_DBN_B0")
            self.assertEqual(dbn._two_tbn.name, "Test_DBN_2TBN")
            #Clean up
            os.remove("/tmp/dbn-test-b0.xbif")
            os.remove("/tmp/dbn-test-2tbn.xbif")
            
    def test_parseDBN_mixed_path(self):
        if os.path.exists("/tmp"):
            import shutil
            shutil.copyfile("primo2/tests/dbn-test-b0.xbif", "/tmp/dbn-test-b0.xbif")
            dbn = DBNSpec.parse("primo2/tests/dbn-test-mixed.conf")
            self.assertEqual(dbn._b0.name, "Test_DBN_B0")
            self.assertEqual(dbn._two_tbn.name, "Test_DBN_2TBN")
            #Clean up
            os.remove("/tmp/dbn-test-b0.xbif")

    def test_parseDBN_load_properties(self):
        dbn = DBNSpec.parse("primo2/tests/dbn-test.conf", ignoreProperties=False)
        aNode = dbn.two_tbn.get_node("A")
        self.assertEqual(len(aNode.meta), 1)
        self.assertTrue("position" in aNode.meta[0])
        
        bNode = dbn.b0.get_node("B")
        self.assertEqual(len(bNode.meta), 1)
        self.assertTrue("position" in bNode.meta[0])
        
    def test_writeDBN(self):
        testPath = "primo2/tests/"
        testName = "test-dbn"
        dbn = DBNSpec.parse("primo2/tests/dbn-test.conf", 
                            ignoreProperties=False)
        
        DBNSpec.write(dbn, testPath, testName) # implicit ignoreProperties=True
        writtenDBN = DBNSpec.parse(testPath+testName+".conf"
                                   ,ignoreProperties=False)
        
        aNode = writtenDBN.two_tbn.get_node("A")
        self.assertEqual(len(aNode.meta), 0)
        
        
        self.assertTrue(len(dbn.b0.get_all_nodes()) == 
                        len(writtenDBN.b0.get_all_nodes()))
        
        self.assertTrue(len(dbn.two_tbn.get_all_nodes()) == 
                        len(writtenDBN.two_tbn.get_all_nodes()))
        
        for node in dbn.b0.get_all_nodes():
            self.assertTrue(node in writtenDBN.b0.get_all_nodes())
            np.testing.assert_array_almost_equal(node.cpd, 
                                        writtenDBN.b0.get_node(node.name).cpd)
            
        for node in dbn.two_tbn.get_all_nodes():
            self.assertTrue(node in writtenDBN.two_tbn.get_all_nodes())
            np.testing.assert_array_almost_equal(node.cpd, 
                                    writtenDBN.two_tbn.get_node(node.name).cpd)
            
        for suf in [".conf", "-b0.xbif", "-2tbn.xbif"]:
            os.remove(testPath + testName + suf)
            
    def test_writeDBN_with_properties(self):
        testPath = "primo2/tests/"
        testName = "test-dbn"
        dbn = DBNSpec.parse("primo2/tests/dbn-test.conf", 
                            ignoreProperties=False)
        DBNSpec.write(dbn, testPath, testName, ignoreProperties=False)
        writtenDBN = DBNSpec.parse(testPath+testName+".conf", 
                                   ignoreProperties=False)
        
        aNode = writtenDBN.two_tbn.get_node("A")
        self.assertEqual(len(aNode.meta), 1)
        
        
        self.assertTrue(len(dbn.b0.get_all_nodes()) == 
                        len(writtenDBN.b0.get_all_nodes()))
        
        self.assertTrue(len(dbn.two_tbn.get_all_nodes()) == 
                        len(writtenDBN.two_tbn.get_all_nodes()))
        
        for node in dbn.b0.get_all_nodes():
            self.assertTrue(node in writtenDBN.b0.get_all_nodes())
            np.testing.assert_array_almost_equal(node.cpd, 
                                        writtenDBN.b0.get_node(node.name).cpd)
            
        for node in dbn.two_tbn.get_all_nodes():
            self.assertTrue(node in writtenDBN.two_tbn.get_all_nodes())
            np.testing.assert_array_almost_equal(node.cpd, 
                                    writtenDBN.two_tbn.get_node(node.name).cpd)
            
        for suf in [".conf", "-b0.xbif", "-2tbn.xbif"]:
            os.remove(testPath + testName + suf)
    
if __name__ == "__main__":
    #Workaround so that this script also finds the resource files when run directly
    # from within the tests folder
    
    os.chdir("../..")
    unittest.main()