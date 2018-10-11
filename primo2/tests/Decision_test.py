#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:47:10 2017

@author: jpoeppel
"""

import unittest

import numpy as np
from primo2.nodes import DecisionNode, UtilityNode, DiscreteNode
from primo2.networks import DecisionNetwork
from primo2.inference.decision import VariableElimination
from primo2.inference.factor import Factor

class VariableEliminationTest(unittest.TestCase):
    
    
    def setUp(self):
        """
        PHD + Startup example 7.4 from Bayesian Reasoning and Machine Learning - Barber
        """
        net = DecisionNetwork()
        education = DecisionNode("education", decisions=["do Phd", "no Phd"]) #E
        startup = DecisionNode("startup", decisions=["start up", "no start up"]) # S
        income = DiscreteNode("income", values=["low", "average", "high"]) #I
        nobel = DiscreteNode("nobel", values=["prize", "no prize"]) #P
        
        costsEducation = UtilityNode("costsE") #UC
        costsStartUp = UtilityNode("costsS") #US
        gains = UtilityNode("gains") #UB
        
        #Add nodes to network. They can be treated the same
        net.add_node(education)
        net.add_node(startup)
        net.add_node(income)
        net.add_node(nobel)
        
        net.add_node(costsEducation)
        net.add_node(costsStartUp)
        net.add_node(gains)
        
        
        #Add edges. Edges can either be acutal dependencies or information links.
        #The type is figured out by the nodes themsevles
        net.add_edge(education, costsEducation)
        net.add_edge(education, nobel)
        
        net.add_edge(startup, income)
        net.add_edge(startup, costsStartUp)
        
        net.add_edge(nobel, income)
        net.add_edge(income, gains)
        
        #Define CPTs: (Needs to be done AFTER the structure is defined as that)
        #determines the table structure for the different nodes
        income.set_probability("low", 0.1, parentValues={"startup":"start up", "nobel":"no prize"})
        income.set_probability("low", 0.2, parentValues={"startup":"no start up", "nobel":"no prize"})
        income.set_probability("low", 0.005, parentValues={"startup":"start up", "nobel":"prize"})
        income.set_probability("low", 0.05, parentValues={"startup":"no start up", "nobel":"prize"})
        
        income.set_probability("average", 0.5, parentValues={"startup":"start up", "nobel":"no prize"})
        income.set_probability("average", 0.6, parentValues={"startup":"no start up", "nobel":"no prize"})
        income.set_probability("average", 0.005, parentValues={"startup":"start up", "nobel":"prize"})
        income.set_probability("average", 0.15, parentValues={"startup":"no start up", "nobel":"prize"})
        
        income.set_probability("high", 0.4, parentValues={"startup":"start up", "nobel":"no prize"})
        income.set_probability("high", 0.2, parentValues={"startup":"no start up", "nobel":"no prize"})
        income.set_probability("high", 0.99, parentValues={"startup":"start up", "nobel":"prize"})
        income.set_probability("high", 0.8, parentValues={"startup":"no start up", "nobel":"prize"})
        
        nobel.set_probability("prize", 0.0000001, parentValues={"education":"no Phd"})
        nobel.set_probability("prize", 0.001, parentValues={"education":"do Phd"})
        
        nobel.set_probability("no prize", 0.9999999, parentValues={"education":"no Phd"})
        nobel.set_probability("no prize", 0.999, parentValues={"education":"do Phd"})
        
        #Define utilities
        costsEducation.set_utility(-50000, parentValues={"education":"do Phd"})
        costsEducation.set_utility(0, parentValues={"education":"no Phd"})
        
        costsStartUp.set_utility(-200000, parentValues={"startup":"start up"})
        costsStartUp.set_utility(0, parentValues={"startup":"no start up"})
        
        gains.set_utility(100000, parentValues={"income":"low"})
        gains.set_utility(200000, parentValues={"income":"average"})
        gains.set_utility(500000, parentValues={"income":"high"})
        
        self.net = net
        self.ve = VariableElimination(net)
        
    def test_combine_factors_chance_nodes(self):
        
        n1 = self.net.node_lookup["income"]
        n2 = self.net.node_lookup["nobel"]
        
        jF1 = Factor.joint_factor(n1)
        jF2 = Factor.joint_factor(n2)
        
        res = self.ve._combine_factors([jF1, jF2])
        #Test correct scope
        self.assertEqual(res[0].variableOrder, ["income", "startup", "nobel", "education"])
        #Test correct potentials using the already tested Factor multiplication
        probFactor = Factor.from_node(n1) * Factor.from_node(n2)
        np.testing.assert_array_almost_equal(res[0].potentials, probFactor.potentials)
        
        np.testing.assert_array_almost_equal(res[1].potentials, np.zeros(probFactor.potentials.shape))
    
    def test_combine_factors_utility_nodes(self):
        
        n1 = self.net.node_lookup["gains"]
        n2 = self.net.node_lookup["costsS"]
        jF1 = Factor.joint_factor(n1)
        jF2 = Factor.joint_factor(n2)
        
        res = self.ve._combine_factors([jF1, jF2])
        self.assertEqual(res[0].variableOrder, ["income", "startup"])
        utFactor = Factor.from_utility_node(n1) + Factor.from_utility_node(n2)
        np.testing.assert_array_almost_equal(res[0].potentials, np.ones(res[0].potentials.shape))
        
        np.testing.assert_array_almost_equal(res[1].potentials, utFactor.potentials)
        
        
    def test_combine_factors_mixed(self):
        n1 = self.net.node_lookup["gains"]
        n2 = self.net.node_lookup["income"]
        jF1 = Factor.joint_factor(n1)
        jF2 = Factor.joint_factor(n2)
        
        res = self.ve._combine_factors([jF1, jF2])
        for var in res[0].variableOrder:
            self.assertIn(var, ["income", "startup", "nobel"])
        
        test_factor = Factor.from_node(n2)
        if res[0].variableOrder == test_factor.variableOrder:
            np.testing.assert_array_almost_equal(res[0].potentials, test_factor.potentials)
        
        np.testing.assert_array_almost_equal(res[1].potentials[:,0,0], Factor.from_utility_node(n1).potentials)
        np.testing.assert_array_almost_equal(res[1].potentials[:,1,0], Factor.from_utility_node(n1).potentials)
        np.testing.assert_array_almost_equal(res[1].potentials[:,0,1], Factor.from_utility_node(n1).potentials)
        np.testing.assert_array_almost_equal(res[1].potentials[:,1,1], Factor.from_utility_node(n1).potentials)
        
        
    def test_marginalize_joint_factor(self):
        n1 = self.net.node_lookup["income"]
        jF1 = Factor.joint_factor(n1)

        res = self.ve._marginalize_joint_factor(jF1, "income")
        self.assertEqual(res[0].variableOrder, ["startup", "nobel"])
        
        probFactor = Factor.from_node(n1).marginalize("income")
        np.testing.assert_array_almost_equal(probFactor.potentials, res[0].potentials)
        self.assertEqual(res[1].potentials.shape, probFactor.potentials.shape)
        
    def test_marginalize_joint_factor_complex(self):
        n1 = self.net.node_lookup["gains"]
        n2 = self.net.node_lookup["income"]
        jF1 = Factor.joint_factor(n1)
        jF2 = Factor.joint_factor(n2)
        
        res = self.ve._combine_factors([jF1, jF2])
        res = self.ve._marginalize_joint_factor(res, "income")
        
        for var in res[0].variableOrder:
            self.assertIn(var, ["startup", "nobel"])
        
        #In this combination of 1 chance and 1 utility node, the probability factor
        #after marginalization is the same as for only the chance node
        probFactor = Factor.from_node(n2)
        marginalizedFactor = probFactor.marginalize("income")
        np.testing.assert_array_almost_equal(res[0].potentials, marginalizedFactor.potentials)
        
        utFactor = Factor.from_utility_node(n1)
        desiredRes = (utFactor * probFactor).marginalize("income") / marginalizedFactor
        np.testing.assert_array_almost_equal(res[1].potentials, desiredRes.potentials)
        
        
    def test_generalized_VE(self):
        n1 = self.net.node_lookup["gains"]
        n2 = self.net.node_lookup["income"]
        jF1 = Factor.joint_factor(n1)
        jF2 = Factor.joint_factor(n2)
        
        res = self.ve.generalized_VE([jF1, jF2], ["income"])
        
        for var in res[0].variableOrder:
            self.assertIn(var, ["startup", "nobel"])
        probFactor = Factor.from_node(n2)
        marginalizedFactor = probFactor.marginalize("income")
        if res[0].variableOrder == marginalizedFactor.variableOrder:
            np.testing.assert_array_almost_equal(res[0].potentials, marginalizedFactor.potentials)
        utFactor = Factor.from_utility_node(n1)
        desiredRes = (utFactor * probFactor).marginalize("income") / marginalizedFactor
        if res[1].variableOrder == desiredRes.variableOrder:
            np.testing.assert_array_almost_equal(res[1].potentials, desiredRes.potentials)
        
    def test_expected_utility(self):
        decisions = {"startup":"no start up", "education":"do Phd"}
        res = self.ve.expected_utility(decisions)
        self.assertEqual(res, 190195)
        
        decisions = {"startup":"no start up", "education":"no Phd"}
        res = self.ve.expected_utility(decisions)
        self.assertTrue(abs(240000.02 - res) < 0.001)
        
        decisions = {"startup":"start up", "education":"do Phd"}
        res = self.ve.expected_utility(decisions)
        self.assertTrue(abs(60186.5 - res) < 0.001)
        
        decisions = {"startup":"start up", "education":"no Phd"}
        res = self.ve.expected_utility(decisions)
        self.assertEqual(res, 110000.01865)
        
        
    def test_get_optimal_decisions(self):
        res = self.ve.get_optimal_decisions(["startup","education"])
        
        desired = {"startup": "no start up", "education":"no Phd"}
        for k,v in res.items():
            self.assertEqual(v, desired[k])
            
            
if __name__ == "__main__":
    unittest.main()