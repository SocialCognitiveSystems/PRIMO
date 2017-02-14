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

import abc
import copy
import six

from . import exact


__all__ = [
    'PriorFeedbackExact',
    'SoftEvidenceExact'
]


@six.add_metaclass(abc.ABCMeta)
class DBNInferenceMethod(object):

    def __init__(self, dbn):
        super(DBNInferenceMethod, self).__init__()
        self._dbn = dbn
        self._t = 0
        self._evidence = False

    @abc.abstractmethod
    def marginal_probabilities(self, variable):
        pass

    @abc.abstractmethod
    def marginals(self, variables):
        pass

    def set_evidence(self, evidence=None, soft_posteriors=False):
        r = self._set_evidence(evidence, soft_posteriors)
        self._evidence = True
        return r

    def unroll(self, evidence=None, soft_posteriors=False):
        if not self._evidence or evidence is not None:
            self._set_evidence(evidence, soft_posteriors)
        r = self._unroll()
        self._evidence = False
        return r

    @abc.abstractmethod
    def _set_evidence(self, evidence=None, soft_posteriors=False):
        pass

    @abc.abstractmethod
    def _unroll(self):
        pass

    @property
    def t(self):
        return self._t


class PriorFeedbackExact(DBNInferenceMethod):

    def __init__(self, dbn):
        super(PriorFeedbackExact, self).__init__(dbn)
        self._dbn = copy.deepcopy(dbn)
        self._ft = None

    def marginal_probabilities(self, variable):
        f = self.marginals([variable])
        return dict(zip(f.values[variable], list(f.get_potential())))

    def marginals(self, variables):
        return self._ft.marginals(variables)

    def _set_evidence(self, evidence=None, soft_posteriors=False):
        if self._t == 0:
            self._ft = exact.FactorTree.create_jointree(self._dbn._b0)
        elif self._t == 1:
            self._ft = exact.FactorTree.create_jointree(self._dbn._two_tbn)
        else:
            self._ft.reset_factors()
        self._ft.set_evidence({} if evidence is None else evidence, softPosteriors=soft_posteriors)

    def _unroll(self):
        for node, node_t in self._dbn._transitions:
            self._dbn._two_tbn.get_node(node).set_cpd(self._ft.marginals([node_t]).get_potential())
        self._t += 1


class SoftEvidenceExact(DBNInferenceMethod):

    def __init__(self, dbn):
        super(SoftEvidenceExact, self).__init__(dbn)
        self._transition_evidence = {}
        self._ft = None

    def marginal_probabilities(self, variable):
        f = self.marginals([variable])
        return dict(zip(f.values[variable], list(f.get_potential())))

    def marginals(self, variables):
        return self._ft.marginals(variables)

    def _set_evidence(self, evidence=None, soft_posteriors=True):
        _evidence = {} if evidence is None else evidence.copy()
        if self._t == 0:
            self._ft = exact.FactorTree.create_jointree(self._dbn._b0)
        else:
            if self._t == 1:
                self._ft = exact.FactorTree.create_jointree(self._dbn._two_tbn)
            _evidence.update(self._transition_evidence)
        self._ft.set_evidence(_evidence, softPosteriors=True)

    def _unroll(self):
        self._transition_evidence.clear()
        for node, node_t in self._dbn._transitions:
            self._transition_evidence[node] = self._ft.marginals([node_t]).get_potential()
        self._t += 1
