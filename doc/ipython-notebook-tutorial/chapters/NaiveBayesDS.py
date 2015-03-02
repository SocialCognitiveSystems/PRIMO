# -*- coding: utf-8 -*-
import numpy



class NaiveBayesDS:
    def __init__(self):
        self.rootNode = None
        self.featureNodes = []
        self.evidence = []

    def calcRootDistribution(self):
        """
        Berechnung der Verteilung des Wurzelknotens.
        P(A | B1, ... Bn) [für alle B mit Evidenz]
        
        A: Wurzelknoten/self.rootNode
        B: featureNode
        """
        # Array für Verteilung anlegen und mit Nullen füllen
        distributionOfA = numpy.zeros(len(self.rootNode[1]))
    
        # Der Divisor für die Formel wird inkrementell aufsummiert
        divisor = 0
      
        # Summe über alle Zustände von A
        for j in range(len(self.rootNode[2])):
       
        # Produkt über alle Evidenzen gegebener Bs, multipliziert mit Aj
            tmpprod = self.rootNode[2][j]
            for i in range(len(self.evidence)):
                tmpNode = self.evidence[i][0]
                evIdx = self.evidence[i][1]
                evIdx = self.getIndex(tmpNode, evIdx)
                evConfidence = self.evidence[i][2]
                
                # Confidence der Evidenz auf CPT rechnen
                tmpCpt = numpy.copy(tmpNode[2])
                ix = 0
                for row in tmpCpt:
                    for ridx in range(len(row)):
                        row[ridx] *= evConfidence
                    remains = 1 - numpy.sum(row)
                    remains /= len(row)
                    for ridx in range(len(row)):
                        row[ridx] += remains
                    tmpCpt[ix] = row
                    ix += 1
                tmpprod *= tmpCpt[j][evIdx]
    
        # Achtung! Hier ist die Verteilung von A noch nicht feritg (s.u.)
            distributionOfA[j] = tmpprod
            divisor += tmpprod
        
        # Abschließend die Verteilung von A durch den Divisor teilen
        distributionOfA = numpy.array(distributionOfA) / divisor
        return distributionOfA

    
    def calcFeatureDistribution(self, featureNode):
        """
        Berechnung der Verteilung eines Feature-Knoten:
        P(B_i | B1, ... B_i-1, B_i+1, ... Bn) [für alle featureNode mit Evidenz]
        
        A: Wurzelknoten/self.rootNode
        B: featureNode
        """
        # Array für Verteilung anlegen und mit Nullen füllen
        distributionOfB = numpy.zeros(len(featureNode[1]))
    
        # Die Verteilung von A kann aus Aufgabe 1 genutzt werden
        distributionOfA = self.calcRootDistribution()
    
        # Verteilung für featureNode iterativ berechnen
        for i in range(len(featureNode[1])):
            tmpsum = 0
            # Multiplikationssatz:
            for j in range(len(distributionOfA)):
                # P(B_i|A_j) * PRODUKT über P(A_j|evidence)
                tmpsum += featureNode[2][j][i] * distributionOfA[j]
            distributionOfB[i] = tmpsum
    
        return distributionOfB
    

    def hasEvidence(self, node):
        """
        Prüft, ob für einen Featureknoten Evidenz gesetzt ist.
        """
        for e in self.evidence:
            enode = e[0]
            if node == enode:
                return True
        return False
            

    def getIndex(self, node, value):
        """
        Index für gegebenen Evidenz-Wert zurückgeben.
        """
        for i in range(len(node[1])):
            if node[1][i] == value:
                return i
        return -1

    def getFeatureEntropy(self, featureNode):
        """
        Entropie für einen Featureknoten berechnen.
        """
        if self.hasEvidence(featureNode):
            return 0
        return self.getEntropy(self.calcFeatureDistribution(featureNode))
    
    def getRootEntropy(self):
        """
        Entropie für den Wurzelknoten berechnen.
        """
        if self.hasEvidence(self.rootNode):
            return 0
        distribution = self.calcRootDistribution()
        return self.getEntropy(distribution)
    
    def getEntropy(self, distribution):
        """
        Entropie für eine Verteilung berechnen.
        
        SUMME über alle Werte der Verteilung mit P(B=b) * log2( P(B=b) )
        """
        entropy = 0.0
        for d in distribution:
            if (d > 0):
                entropy -= d * numpy.log2(d)
        return entropy
    
    
    def getConditionalEntropy(self, featureNode):
        """
        Bedingte Entropie für den Wurzelknoten bei gegebenen Featureknoten
        berechnen, für welchen noch keine Evidenz vorliegt.
        
        SUMME über alle b aus B mit P(B=b) * H(A|B=b)
        """
        entropy = 0.0
        if self.hasEvidence(featureNode):
            return 0
        
        idx = 0
        for value in featureNode[1]:
            # P(B=b)
            distB = self.calcFeatureDistribution(featureNode)
            probValue = distB[idx]
            
            # H(A|B=b)
            evEntry = [featureNode, value, 1.0]
            self.evidence.append(evEntry)
            rootEntropy = self.getRootEntropy()
            self.evidence = self.evidence[:-1]
            
            # SUMME += P(B=b) * H(A|B=b)
            entropy += probValue * rootEntropy
            
            idx += 1
            
        return entropy





