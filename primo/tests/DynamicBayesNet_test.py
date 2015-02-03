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

import unittest

from primo.networks import DynamicBayesianNetwork
from primo.nodes import DiscreteNode


class DynamicBayesNetTest(unittest.TestCase):
    def setUp(self):
        self.dbn = DynamicBayesianNetwork()

    def tearDown(self):
        self.dbn = None

    def test_add_node(self):
        self.dbn.clear()
        n = DiscreteNode("Some_Node", [True, False])
        self.dbn.add_node(n)
        self.assertEqual(n, self.dbn.get_node("Some_Node"))
        self.assertTrue(n in self.dbn.get_nodes(["Some_Node"]))

    def test_temporal_edges(self):
        self.dbn.clear()
        n1 = DiscreteNode("1", [True, False])
        n2 = DiscreteNode("2", [False, False])
        self.dbn.add_node(n1)
        self.dbn.add_node(n2)
        self.assertTrue(self.dbn.is_valid())
        self.dbn.add_edge(n1, n1)
        self.assertFalse(self.dbn.is_valid())
        self.dbn.remove_edge(n1, n1)
        self.dbn.add_edge(n1, n2)
        self.assertTrue(self.dbn.is_valid())


#include this so you can run this test without nose
if __name__ == '__main__':
    unittest.main()
