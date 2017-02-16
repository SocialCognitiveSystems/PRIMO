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
    """Abstract class for dynamic Bayesian network inference methods.

    Contains the inference algorithm and the current state of the 
    network at time t.
    """

    def __init__(self, dbn):
        """Create the inference method and its state.

        Parameters
        ----------
        dbn : DynamicBayesianNetwork
            A 2-TBN dynamic Bayesian network structure.
        """
        super(DBNInferenceMethod, self).__init__()
        self._dbn = dbn
        self._t = 0
        self._evidence = False

    @abc.abstractmethod
    def marginal_probabilities(self, variable):
        """Get the marginal probabilities of a variable. 

        This is an abstract method that needs to be implemented by subclasses
        of `DBNInferenceMethod`.

        Parameters
        ----------
        variable : RandomNode|String
            The variable/node.

        Returns
        -------
        dict
            A dictionary, dict-keys are the values of the variable/node and 
            dict-values are the probabilities of each value.
        """
        pass

    @abc.abstractmethod
    def marginals(self, variables):
        """Get the marginals of a variable.

        This is an abstract method that needs to be implemented by subclasses
        of `DBNInferenceMethod`.

        Parameters
        ----------
        variables : [String,]
            List containing the names of the variables whose joint prior
            or posterior marginals are desired.   
                
        Returns
        -------
        ?
            The marginals.
        """
        pass

    def set_evidence(self, evidence=None, **kwargs):
        """Set evidence for the current time slice, without proceeding in time.

        Calls the method `_set_evidence`, which does the actual work.
        
        Parameters
        ----------
        evidence : dict, optional
            Dictionary containing the given evidence. The keys represent the
            variable names of the given evidence and the values can either
            be simple strings of the given values, or a np.array specifying
            the strength of the evidence for each outcome, e.g.: For the binary
            evidence node E `{"E": "True"}` is equivalent to
            `{"E": np.array([1.0, 0.0])}`.
        **kwargs
            Takes the same keyword arguments as `set_evidence` and
            `_set_evidence`. These are additional argument that derivatives
            of this class can use.
        """
        r = self._set_evidence(evidence)
        self._evidence = True
        return r

    def unroll(self, evidence=None, **kwargs):
        """Unroll the network, and proceed in time.

        If no evidence has been set (via `set_evidence`) or is specified in the
        arguments it proceeds without evidence. If evidence has been set
        previously and no evidence is specified in the arguments it uses this
        pre-specified evidence. If evidence is set in the arguments, this new
        evidence is used.

        Calls the method `_unroll`, which does the actual work.

        evidence : dict, optional
            Dictionary containing the given evidence. The keys represent the
            variable names of the given evidence and the values can either
            be simple strings of the given values, or a np.array specifying
            the strength of the evidence for each outcome, e.g.: For the binary
            evidence node E `{"E": "True"}` is equivalent to
            `{"E": np.array([1.0, 0.0])}`.
        **kwargs
            Takes the same keyword arguments as `set_evidence` and
            `_set_evidence`. These are additional argument that derivatives
            of this class can use.
        """
        if not self._evidence or evidence is not None:
            self._set_evidence(evidence, **kwargs)
        r = self._unroll()
        self._evidence = False
        return r

    @abc.abstractmethod
    def _set_evidence(self, evidence=None, **kwargs):
        """Set evidence for the network (doing the actual work).

        This is an abstract method that needs to be implemented by subclasses
        of `DBNInferenceMethod`.

        See `set_evidence`.
        """
        pass

    @abc.abstractmethod
    def _unroll(self):
        """Unroll the network (doing the actual work).

        This is an abstract method that needs to be implemented by subclasses
        of `DBNInferenceMethod`.

        See `unroll`.
        """
        pass

    @property
    def t(self):
        """Get number current time-slice.

        Returns
        -------
        int
            The number of the current time-slice.
        """
        return self._t


class PriorFeedbackExact(DBNInferenceMethod):
    """Prior feedback-based approach to unrolling a DBN, using exact inference.

    Inference is exact and based on factor trees. The marginal posterior
    probabilities of the variables of the current time-slice are used as prior
    probabilties for the next time-slice. They are set as the conditional
    probability distribution of the nodes/variables.
    """

    def __init__(self, dbn):
        super(PriorFeedbackExact, self).__init__(dbn)
        self._dbn = copy.deepcopy(dbn)
        self._ft = None

    def marginal_probabilities(self, variable):
        """Get the marginal probabilities of a variable.

        See `DBNInferenceMethod.marginal_probabilities`.
        """ 
        f = self.marginals([variable])
        return dict(zip(f.values[variable], list(f.get_potential())))

    def marginals(self, variables):
        """Get the marginals of a variable.

        Parameters
        ----------
        variables : [String,]
            List containing the names of the variables whose joint prior
            or posterior marginals are desired.
            Joint marginals for multiple variables can only be computed
            if they are contained in any clique.     
                
        Returns
        -------
        Factor
            A factor containing the desired marginals.
        """
        return self._ft.marginals(variables)

    def _set_evidence(self, evidence=None, soft_posteriors=False):
        """Set evidence for the network (doing the actual work).

        Parameters
        ----------
        evidence : dict
            See `DBNInferenceMethod.set_evidence`.
        softPosteriors: bool, optional
            If softPosterior is set to True, the evidence, should it be an
            np.array, is interpreted as soft evidence for a desired posterior
            distribution. See `primo2.exact.FactorTree.set_evidence`.
        """
        if self._t == 0:
            self._ft = exact.FactorTree.create_jointree(self._dbn._b0)
        elif self._t == 1:
            self._ft = exact.FactorTree.create_jointree(self._dbn._two_tbn)
        else:
            self._ft.reset_factors()
        self._ft.set_evidence({} if evidence is None else evidence, softPosteriors=soft_posteriors)

    def _unroll(self):
        """Unroll the network, and proceed in time.

        See `DBNInferenceMethod.unroll`.
        """
        for node, node_p in self._dbn._transitions:
            self._dbn._two_tbn.get_node(node).set_cpd(
                    self._ft.marginals([node_p]).get_potential())
        self._t += 1


class SoftEvidenceExact(DBNInferenceMethod):
    """Prior feedback-based approach to unrolling a DBN, using exact inference.

    Inference is exact and based on factor trees. The marginal posterior
    probabilities of the variables of the current time-slice are used as prior
    probabilties for the next time-slice. They are set as soft evidence on the
    FactorTree (see `factorTree.set_evidence`).
    """

    def __init__(self, dbn):
        super(SoftEvidenceExact, self).__init__(dbn)
        self._transition_evidence = {}
        self._ft = None

    def marginal_probabilities(self, variable):
        """Get the marginal probabilities of a variable.

        See `DBNInferenceMethod.marginal_probabilities`.
        """ 
        f = self.marginals([variable])
        return dict(zip(f.values[variable], list(f.get_potential())))

    def marginals(self, variables):
        """Get the marginals of a variable.

        Parameters
        ----------
        variables : [String,]
            List containing the names of the variables whose joint prior
            or posterior marginals are desired.
            Joint marginals for multiple variables can only be computed
            if they are contained in any clique.     
                
        Returns
        -------
        Factor
            A factor containing the desired marginals.
        """
        return self._ft.marginals(variables)

    def _set_evidence(self, evidence=None):
        """Set evidence for the network (doing the actual work).

        Automatically set `softPosteriors` to `True` as prior feedback is fed
        into the network as soft evidence.

        Parameters
        ----------
        evidence : dict
            See `DBNInferenceMethod.set_evidence`.
        """
        _evidence = {} if evidence is None else evidence.copy()
        if self._t == 0:
            self._ft = exact.FactorTree.create_jointree(self._dbn._b0)
        else:
            if self._t == 1:
                self._ft = exact.FactorTree.create_jointree(self._dbn._two_tbn)
            _evidence.update(self._transition_evidence)
        self._ft.set_evidence(_evidence, softPosteriors=True)

    def _unroll(self):
        """Unroll the network, and proceed in time.

        See `DBNInferenceMethod.unroll`.
        """
        self._transition_evidence.clear()
        for node, node_p in self._dbn._transitions:
            self._transition_evidence[node] = \
                    self._ft.marginals([node_p]).get_potential()
        self._t += 1
