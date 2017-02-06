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

import numpy as np

from primo2.nodes import DiscreteNode
from primo2.inference.factor import Factor


wet = DiscreteNode("wet_grass")
sprink = DiscreteNode("sprinkler")
rain = DiscreteNode("rain")
winter = DiscreteNode("winter")
slippery = DiscreteNode("slippery_road")
slippery.add_parent(rain)
wet.add_parent(sprink)
wet.add_parent(rain)
sprink.add_parent(winter)
rain.add_parent(winter)

winter.set_cpd(np.array([0.6,0.4]))
print "winter: ", winter.cpd
rain.set_cpd(np.array([[0.8, 0.1],[0.2,0.9]]))
print "rain: ", rain.cpd
sprink.set_cpd(np.array([[0.2,0.75],[0.8,0.25]]))
print "sprinkler: ", sprink.cpd
slippery.set_cpd(np.array([[0.7,0.0],[0.3,1.0]]))
print "slippery: ", slippery.cpd
wet.set_cpd(np.array([[[0.95, 0.1],[0.8,0.0]],[[0.05,0.9],[0.2,1.0]]]))
print "wet: ", wet.cpd

print "wet prob: ", wet.get_probability("False", {"sprinkler":["True"], "rain":["True","False"]})

#import sys
#sys.exit()
f_sl = Factor.from_node(slippery)
f_wg = Factor.from_node(wet)
f_ev = Factor.as_evidence("wet_grass", ["True","False"], "False")
f_sp = Factor.from_node(sprink)
f_w = Factor.from_node(winter)
f_r = Factor.from_node(rain)

f_wgE = f_wg * f_ev

#print "potentials: ", f_wgE.get_potential({"wet_grass":["False"], "rain":["True", "False"], "sprinkler":["True","False"]})
#print f_wgE.potentials
#print f_wgE.marginalize("wet_grass").potentials
f_wgEm = f_wgE.marginalize("wet_grass")

#print (f_sp * f_wgEm).potentials
#print (f_sp * f_wgEm).marginalize("sprinkler").potentials

print "f_wgEm variableorder: ", f_wgEm.variableOrder
print "f_wgEm variables: ", f_wgEm.variables
print "f_wgEm: ", f_wgEm.potentials
#print "f_sp: ", f_sp.potentials
print "sp*wg: ", (f_sp * f_wgEm).potentials
#f_spM = (f_sp * f_wgEm).marginalize("sprinkler")
#print "spM: ", f_spM.potentials
#f_rw = f_r * f_w
#
##print "rain*winter: ", (f_r * f_w ).potentials
#
#print "spM*rw: ", (f_spM * f_rw).marginalize("winter").potentials
#
#res = (f_spM * f_rw).marginalize("winter")
#res.normalize()
#
#print res.potentials
#
##print "sl marg: ", f_sl.marginalize("slippery_road").potentials