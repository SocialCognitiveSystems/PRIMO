#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:59:32 2017

@author: jpoeppel
"""

from primo2.networks import DecisionNetwork
from primo2.nodes import DiscreteNode, DecisionNode, UtilityNode

from primo2.inference.decision import VariableElimination

"""
PHD example 7.3 from Bayesian Reasoning and Machine Learning - Barber
"""
net = DecisionNetwork()

education = DecisionNode("education", decisions=["do Phd", "no Phd"]) #E

income = DiscreteNode("income", values=["low", "average", "high"]) #I
nobel = DiscreteNode("nobel", values=["prize", "no prize"]) #P

costs = UtilityNode("costs") #UC
gains = UtilityNode("gains") #UB


#Add nodes to network. They can be treated the same
net.add_node(education)
net.add_node(income)
net.add_node(nobel)

net.add_node(costs)
net.add_node(gains)


#Add edges. Edges can either be acutal dependencies or information links.
#The type is figured out by the nodes themsevles
net.add_edge(education, costs)
net.add_edge(education, nobel)
net.add_edge(education, income)

net.add_edge(nobel, income)
net.add_edge(income, gains)

#Define CPTs: (Needs to be done AFTER the structure is defined as that)
#determines the table structure for the different nodes

income.set_probability("low", 0.1, parentValues={"education":"do Phd", "nobel":"no prize"})
income.set_probability("low", 0.2, parentValues={"education":"no Phd", "nobel":"no prize"})
income.set_probability("low", 0.01, parentValues={"education":"do Phd", "nobel":"prize"})
income.set_probability("low", 0.01, parentValues={"education":"no Phd", "nobel":"prize"})

income.set_probability("average", 0.5, parentValues={"education":"do Phd", "nobel":"no prize"})
income.set_probability("average", 0.6, parentValues={"education":"no Phd", "nobel":"no prize"})
income.set_probability("average", 0.04, parentValues={"education":"do Phd", "nobel":"prize"})
income.set_probability("average", 0.04, parentValues={"education":"no Phd", "nobel":"prize"})

income.set_probability("high", 0.4, parentValues={"education":"do Phd", "nobel":"no prize"})
income.set_probability("high", 0.2, parentValues={"education":"no Phd", "nobel":"no prize"})
income.set_probability("high", 0.95, parentValues={"education":"do Phd", "nobel":"prize"})
income.set_probability("high", 0.95, parentValues={"education":"no Phd", "nobel":"prize"})


nobel.set_probability("prize", 0.0000001, parentValues={"education":"no Phd"})
nobel.set_probability("prize", 0.001, parentValues={"education":"do Phd"})

nobel.set_probability("no prize", 0.9999999, parentValues={"education":"no Phd"})
nobel.set_probability("no prize", 0.999, parentValues={"education":"do Phd"})


#Define utilities

costs.set_utility(-50000, parentValues={"education":"do Phd"})
costs.set_utility(0, parentValues={"education":"no Phd"})

gains.set_utility(100000, parentValues={"income":"low"})
gains.set_utility(200000, parentValues={"income":"average"})
gains.set_utility(500000, parentValues={"income":"high"})


ve = VariableElimination(net)

print("Expected Utility for doing a Phd: {}".format(ve.expected_utility(decisions={"education":"do Phd"})))
print("Expected Utility for not doing a Phd: {}".format(ve.expected_utility(decisions={"education":"no Phd"})))

print("Optimal deciosn: ", ve.get_optimal_decisions(["education"]))




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

ve = VariableElimination(net)

print("Expected Utility for doing a Phd + startup: {}".format(ve.expected_utility(decisions={"education":"do Phd", "startup": "start up"})))
print("Expected Utility for doing a Phd + no startup: {}".format(ve.expected_utility(decisions={"education":"do Phd", "startup": "no start up"})))
print("Expected Utility for not doing a Phd + startup: {}".format(ve.expected_utility(decisions={"education":"no Phd", "startup": "start up"})))
print("Expected Utility for not doing a Phd + no startup: {}".format(ve.expected_utility(decisions={"education":"no Phd", "startup": "no start up"})))


print("Optimal deciosn: ", ve.get_optimal_decisions(["startup", "education"]))