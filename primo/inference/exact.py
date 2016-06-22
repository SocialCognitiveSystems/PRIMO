#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:52:59 2016

@author: jpoeppel
"""
from __future__ import division 

import networkx as nx

from primo.inference.factor import Factor
from primo.inference.order import Orderer

class VariableElimination(object):
    
    
    @staticmethod
    def naive_marginals(bn, variables, evidence=None):
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
        
        if not evidence:
            evidence = {}
        
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
    def bucket_marginals(bn, variables, evidence=None, order=None):
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
            
        if not evidence:
            evidence = {}
        
        #Move query variables at the end of the order
        for v in variables:
            order.remove(v)
            order.append(v)
            
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
        for e in evidence.keys():
            bucketI = order.index(e)
            buckets[bucketI] = buckets[bucketI] * Factor.as_evidence(e, bn.get_node(e).values, evidence[e])
            
        bucketUntil = len(order)- len(variables)
        # Process buckets
        for i in range(bucketUntil):
            tmpFactor = buckets[i].marginalize(order[i])
            for j in range(i+1, len(order)):
                if order[j] in tmpFactor.variables:
                    buckets[j] = buckets[j] * tmpFactor
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
    
    def __init__(self, tree, bn):
        self.tree = tree
        self.bn = bn
        
    
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
        factors = []
        #Add edges between parents
        for n in bn.get_all_nodes():
            factors.append(Factor.from_node(n))
            for p in n.parents:
                for p2 in n.parents:
                    if p != p2:
                        moralG.add_edge(n.parents[p], n.parents[p2])
        
        #Determine cluster sequence
        clusterSeq = []
        for v in order:
            variables = [v]+[n.name for n in moralG.neighbors(v)]
            clusterSeq.append(variables)
            moralG.remove_node(v)
        swap= []
        #Remove nonmaximal clusters
        for i in range(len(clusterSeq)-1,0,-1):
            for j in range(i-1,-1,-1):
                if set(clusterSeq[i]).issubset(set(clusterSeq[j])):
                    swap.append((clusterSeq[i], clusterSeq[j]))
                    break
        #Remove cluster i and place cluster j at it's location     
        for rem, move in swap:
            idx = clusterSeq.index(rem)
            clusterSeq.remove(rem)
            clusterSeq.insert(idx, move)
            clusterSeq.remove(move)
        # Construct jointree
        tree = nx.Graph(messagesValid=False)
        if len(clusterSeq) > 0:
            tree.add_node("".join(clusterSeq[-1]), variables=set(clusterSeq[-1]), factor=Factor.get_trivial())
            for i in range(len(clusterSeq)-2,-1,-1):
                tree.add_node("".join(clusterSeq[i]), variables=set(clusterSeq[i]), factor=Factor.get_trivial())
                jointreeProp = set(clusterSeq[i]).intersection(set().union(*clusterSeq[i+1:]))
                for cl in clusterSeq[i+1:]:
                    if jointreeProp.issubset(set(cl)):
                        tree.add_edge("".join(clusterSeq[i]), "".join(cl), sep=jointreeProp, factor=Factor.get_trivial())
                        break
                    
        # Assign factors to clusters
        for f in factors:
            for treeNode, treeData in tree.nodes_iter(data=True):
                if set(f.variables).issubset(treeData["variables"]):
                    treeData["factor"] = treeData["factor"] * f
                    break
        return cls(tree,bn)
        

    def reset_factors(self):
        """
            Resets all the factors in the jointree to the standards given by the
            BayesianNetwork. This is necessary when the evidence changes, or maybe
            some probabilities changed within the BayesianNetwork.
        """        
        
        #Create new factors
        factors = []
        for n in self.bn.get_all_nodes():
            factors.append(Factor.from_node(n))
        #Reset factors in nodes and edges of the tree
        for treeNode, treeData in self.tree.nodes_iter(data=True):
            treeData["factor"] = Factor.get_trivial()
        for u,b, edgeData in self.tree.edges_iter(data=True):
            edgeData["factor"] = Factor.get_trivial()
            
        for f in factors:
            for treeNode, treeData in self.tree.nodes_iter(data=True):
                if set(f.variables).issubset(treeData["variables"]):
                    treeData["factor"] = treeData["factor"] * f
                    break
        self.tree.graph["messagesValid"] = False
        
        
    def set_evidence(self, evidence):
        """
            Sets the given evidence in the factor tree. This will trigger a 
            recomputation of the messages given this evidence.
            
            Parameters
            ----------
            evidence : dict, optional
                Dictionary containing the given evidence. The keys represent the
                variable names of the given evidence and the values can either
                be simple strings of the given values, or a np.array specifying
                the strength of the evidence for each outcome, e.g.:
                For the binary evidence node E
                evidence = {"E": "True"} is equivalent to {"E": np.array([1.0,0.0])}
        """
        self.reset_factors()
        # Add evidence to buckets
        for e in evidence.keys():
            evidenceFactor = Factor.as_evidence(e, self.bn.get_node(e).values, evidence[e])
            for node, nodeData in self.tree.nodes_iter(data=True):
                if e in nodeData["variables"]:
                    nodeData["factor"] = nodeData["factor"] * evidenceFactor
                    break
        self.calculate_messages()
        
    def marginals(self, variables):
        """
            Function to compute marginals for the given variables, potentially
            given some evidence that was set beforehand using set_evidence.
            The marginals can be computed efficiently using the Hugin
            architecture (see "Modeling and Reasoning with Bayesian Networks" - 
            Adnan Darwiche Chapter 7 and especially 7.7.4)

            
            Parameters
            ----------
            variables : [String,]
                List containing the names of the variables whose joint prior
                or posterior marginals are desired.
                Joint marginals for multiple variables can only be computed
                if they are contained in any clique. To compute the joint
                marginals for a fixed instantiation, this instantiation can
                be set as evidence and its probability can be queried using
                get_evidence_probability()                
                
            Returns
            -------
                Factor
                A factor containing the desired marginals
        """
        if not self.tree.graph["messagesValid"]:
            self.calculate_messages()
            
        # Determine clique containing variables:
        varSet = set(variables)
        for treeNode, treeData in self.tree.nodes_iter(data=True):
            if varSet.issubset(treeData["variables"]):
                resFactor = treeData["factor"].marginalize(treeData["variables"] - varSet)
                resFactor.normalize()
                return resFactor
        else:
            # No suitable clique found
            raise ValueError("No clique containing the variables {} was found.".format(variables))
            
        
    def get_evidence_probability(self):
         raise NotImplementedError("We still need to implement this...")
        
    def calculate_messages(self):
        """
            Performs the two way (inward and outward) message passing with
            the first node as root. Is needed to validate the messages in the
            jointree.
        """
        try:
            root = self.tree.nodes()[0]
            self.pull_messages(self.tree, root, None)
            self.push_messages(self.tree, root, None)
            self.tree.graph["messagesValid"] = True
        except IndexError:
            pass
        
    def pull_messages(self, tree, curNode, parent):
        """
            Performs the inward message passing from the given node to its
            parent according to Hugin's architecture.
            
            Parameters
            ----------
            tree : nx.Graph
                The underlying jointree. Is only passed as parameter for slight
                performance improvements.
            curNode : String
                Name of the clique node within the jointree that should currently
                send it's message to the given parent.
            parent : String
                Name of the parent node the message should be passed to.
        """
        #Let neighbors collect messages
        for neighbor in tree.neighbors_iter(curNode):
            if neighbor != parent:
                self.pull_messages(tree, neighbor, curNode)
                      
        # Send message to parent
        if parent:
            newSeqFactor = tree.node[curNode]["factor"].marginalize(tree.node[curNode]["variables"]-tree[curNode][parent]["sep"])
            tree.node[parent]["factor"] = tree.node[parent]["factor"] * (newSeqFactor / tree[curNode][parent]["factor"])
            tree[curNode][parent]["factor"] = newSeqFactor
        else:
            return
            
    def push_messages(self, tree, curNode, parent):
        """
            Performs the outwards message passing from the given node to its
            children other than the given parent according to Hugin's architecture.
            
            Parameters
            ----------
            tree : nx.Graph
                The underlying jointree. Is only passed as parameter for slight
                performance improvements.
            curNode : String
                Name of the clique node within the jointree that should currently
                send it's message to it's children
            parent : String
                Name of the parent node to avoid sending messages back to the
                parent.
        """
        for neighbor in tree.neighbors_iter(curNode):
            if neighbor != parent:
                #Send message out to neighbor
                newSeqFactor = tree.node[curNode]["factor"].marginalize(tree.node[curNode]["variables"]-tree[curNode][neighbor]["sep"])
                tree.node[neighbor]["factor"] = tree.node[neighbor]["factor"] * (newSeqFactor / tree[curNode][neighbor]["factor"])
                tree[curNode][neighbor]["factor"] = newSeqFactor
                # Have neighbor pushing out further
                self.push_messages(tree, neighbor, curNode)
            
        