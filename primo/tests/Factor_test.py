#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:46:05 2016

@author: jpoeppel
"""

import unittest
import numpy as np

class FactorTest(unittest.TestCase):


    def setUp(self):
        cptA = np.array([0.6,0.4])
        cptB_A = np.array([[0.9, 0.2],[0.1,0.8]])
        cptC_B = np.array([[0.3,0.5], [0.7,0.5]])

    def test_multiplication(self):
        f1 = Factor()
        f1.table = cptA
        f1.var_index = {"A":0}
        f1.val_index = {"A": {"True":0, "False":1}}
        f2 = Factor()
        f2.table = cptB_A
        f2.var_index = {"B": 0, "A":1}
        f2.val_index = {"B": {"True":0, "False":1}, "A":{"True":0,"False":1}}
        res = np.array([[0.54,0.08], [0.06,0.32]])
        f3 = f1 * f2
        np.testing.assert_array_equal(f3.table, res)
        
        
    def test_marginalisation(self):
        self.fail("TODO")
        
    def test_division(self):
        self.fail("TODO")
    
    
if __name__ == "__main__":
    unittest.main()