#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:06:58 2016

@author: jpoeppel
"""

import numpy as np

class Factor(object):
    """
        Class representing a factor in an inference network.
        Factors can be multiplied together and variables can be summed out
        (marginalised) of factors.
    """
    
    def __init__(self):
        self.variables = []
        self.table = np.zeros()
        self.var_index = {}
        self.val_index = {}