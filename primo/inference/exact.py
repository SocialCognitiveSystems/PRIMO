#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:52:59 2016

@author: jpoeppel
"""
import networkx as nx

from primo.inference.factor import Factor
from primo.inference.order import Orderer

class VariableElimination(object):
    
    
    @staticmethod
    def naive_marginals(bn, variables, evidence={}):
        """
            Function to compute the prior or posterior marginals given evidence
            from a given Bayesian Network for the variables and their 
            values of interest. Uses the naive approach of multiplying all 
            variables together before marginalizing the undesired variables.
            
            Parameters
            ----------
            bn : BayesianNetwork
                The network that is supposed to be queried.
            variables : [String,]
                List containing the names of the variables whose joint prior
                or posterior marginals are desired.
            evidence : dict, optional
                Dictionary containing the given evidence. The keys represent the
                variable names of the given evidence and the values can either
                be simple strings of the given values, or a np.array specifying
                the strength of the evidence for each outcome, e.g.:
                For the binary evidence node E
                evidence = {"E": "True"} is equivalent to {"E": np.array([1.0,0.0])}
                
            Returns
            -------
                Factor
                A factor containing the desired marginals
        """
        
        order = bn.get_all_node_names()
        # Compute joint probability distribution of all variables
        resFactor = Factor.get_trivial()            
        for v in order:
            resFactor = resFactor * Factor.from_node(bn.get_node(v))
            #Add evidence as additional factors
            if v in evidence:
                resFactor = resFactor * Factor.as_evidence(v, bn.get_node(v).values, evidence[v])
                
        
        # Marginalise unwated variables
        for v in order:
            if v not in variables:
                resFactor = resFactor.marginalize(v)
                
        # Normalize to get conditional probability for the evidence
        resFactor.normalize()
                
        return resFactor
        
    
    @staticmethod
    def bucket_marginals(bn, variables, evidence={}, order=None):
        """
            Function to compute the prior or posterior marginals given evidence
            from a given Bayesian Network for the variables and their 
            values of interest. Uses the bucket elimination strategy proposed
            in "Modeling and Reasoning with Bayesian Networks" - Adnan Darwiche
            Chapter 6.10
            
            Parameters
            ----------
            bn : BayesianNetwork
                The network that is supposed to be queried.
            variables : [String,]
                List containing the names of the variables whose joint prior
                or posterior marginals are desired.
            evidence : dict, optional
                Dictionary containing the given evidence. The keys represent the
                variable names of the given evidence and the values can either
                be simple strings of the given values, or a np.array specifying
                the strength of the evidence for each outcome, e.g.:
                For the binary evidence node E
                evidence = {"E": "True"} is equivalent to {"E": np.array([1.0,0.0])}
            order : [String,], optional
                List containing the elimination order of the nodes. If order is not
                given, this algorithm computes the min degree order automatically.
                
            Returns
            -------
                Factor
                A factor containing the desired marginals
        """
        
        if not order:
            order = Orderer.get_min_degree_order(bn)
        
        #Move query variables at the end of the order
        for v in variables:
            order.remove(v)
            order.append(v)
            
#        print "order: ", order
            
        factors = [Factor.from_node(bn.get_node(v)) for v in order]
        
        #Create buckets with trivial factors (one more than variables
        #for trivial factors later
        buckets = [Factor.get_trivial() for i in range(len(order)+1)]
        
        #Place factors in the first buckets they correlate to
        for f in factors:
            for i in range(len(order)):
                if order[i] in f.variables:
                    buckets[i] = buckets[i] * f
                    break                                        

        # Add evidence to buckets
        for e in evidence.iterkeys():
            bucketI = order.index(e)
            buckets[bucketI] = buckets[bucketI] * Factor.as_evidence(e, bn.get_node(e).values, evidence[e])
#            print "evidence for: ", e
#            print "resulting bucket table: ", buckets[bucketI].potentials
#            print "variable order: ", buckets[bucketI].variableOrder
            
        bucketUntil = len(order)- len(variables)
        # Process buckets
        for i in range(bucketUntil):
#            print "margenilize: ", order[i]
            tmpFactor = buckets[i].marginalize(order[i])
#            print "tmpFactor.variables: ", tmpFactor.variables
#            print "tmpFactor.table: ", tmpFactor.potentials
            for j in range(i+1, len(order)):
                if order[j] in tmpFactor.variables:
#                    print "inserting in bucket: ", order[j]
#                    print "bucket before: ", buckets[j].potentials
                    buckets[j] = buckets[j] * tmpFactor
#                    print "resulting bucket: ", buckets[j].potentials
                    break
            else:
                # No suitable bucket found -> factor is trivial
                buckets[-1] = buckets[-1] * tmpFactor
                
        #Compute marginals of intended variables
        for i in range(bucketUntil, len(buckets)-1):
            buckets[i+1] = buckets[i+1] * buckets[i]
            
        # Normalize evidence
        buckets[-1].normalize()

        return buckets[-1]
        
class FactorTree(object):
    
    def __init__(self, tree):
        self.tree = tree
        
    
    @classmethod
    def create_jointree(cls, bn, order=None):
        """
            Creates a jointree according to 
            "Modeling and Reasoning with Bayesian Networks" - Adnan Darwiche
            Chapter 9.4.1
            
            Parameter
            ---------
            bn : BayesianNetwork
                The network that is used to create the jointree
                
            order : [String,], optional
                Elimination order used to create the jointree. If order is not
                given, this algorithm computes the min degree order automatically.
                
            Returns
            -------
                FactorTree
                The jointree/factortree that can be used to query the network
                efficiently.
        """
        
        if not order:
            order = Orderer.get_min_degree_order(bn)
        moralG = bn.graph.to_undirected()
        #Add edges between parents
        for n in bn.get_all_nodes():
            for p in n.parents:
                for p2 in n.parents:
                    if p != p2:
                        moralG.add_edge(n.parents[p], n.parents[p2])
        #Determine cluster sequence
        clusterSeq = []
        for v in order:
            variables = [v]+[n.name for n in moralG.neighbors(v)]
            clusterSeq.append(str(variables))
            moralG.remove_node(v)
        moveTo = []
        #Remove nonmaximal clusters
        for i in range(len(clusterSeq)-1,0,-1):
            for j in range(i-1,-1,-1):
                if set(clusterSeq[i]).issubset(set(clusterSeq[j])):
#                    print "cluster {}({}) is subset of cluster {}({})".format(i,clusterSeq[i], j,clusterSeq[j])
                    moveTo.append((i,j))
                    break
        #Remove cluster i and place cluster j at it's location                    
        for i,j in moveTo:
#            print i,j
            tmp = clusterSeq[j]
            clusterSeq.remove(clusterSeq[i])
            clusterSeq.insert(i, tmp)
            clusterSeq.remove(tmp)
            
#        print "order: ", order
#        print clusterSeq
        
        # Construct jointree
        tree = nx.Graph()
        tree.add_node("".join(clusterSeq[-1]))
        for i in range(len(clusterSeq)-2,-1,-1):
            tree.add_node(clusterSeq[i])
            jointreeProp = set(clusterSeq[i]).intersection(set(clusterSeq[i+1:]))
            for cl in clusterSeq[i+1:]:
                if jointreeProp.issubset(set(cl)):
                    tree.add_edge("".join(clusterSeq[i]), "".join(cl))
                    
        return cls(tree)
            
        