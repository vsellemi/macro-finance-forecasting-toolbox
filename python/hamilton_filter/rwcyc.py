#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 08:51:02 2021

@author: victorsellemi
"""

import numpy as np

def rwcyc(y):  
    """
    Python code to calculate cyclical component based on 2-year-ahead forecast error for baseline case of random walk as recommended in
       James D. Hamilton, "Why You Should Never Use the Hodrick-Prescott Filter"
       Review of Economics and Statistics
    input:  y = (T x 1) vector of data, tth element is observation for date t
    output  ydif = (T x 1) vector, tth element is cyclical component for date t

    """
    
    T = y.shape[0]
    y = y.reshape((T,1))
    ydif = np.nan * np.ones((T,1))
    
    h = 8 # default for quarterly data and 2-year horizon
    
    ydif[h:] = y[h:] - y[:(T-h)]
    
    return ydif
