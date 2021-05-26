#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 08:51:02 2021

@author: victorsellemi
"""

import numpy as np
import numpy.linalg as LA

def regcyc(y):  
    """
    Python code to calculate cyclical component based on 2-year forecast error
    from linear regression as recommended in
        James D. Hamilton, "Why You Should Never Use the Hodrick-Prescott Filter"
        Review of Economics and Statistics
    input:  y = (T x 1) vector of data, tth element is observation for date t
    output  yreg = (T x 1) vector, tth element is cyclical component for date t

    """
    
    T = y.shape[0]
    y = y.reshape((T,1))
    yreg = np.nan * np.ones((T,1))
    
    h = 8 # default for quarterly data and 2-year horizon
    p = 4 # default for quarterly data (number of lags in regression)
    
    # construct X matrix of lags
    X = np.ones((T-p-h+1,1))
    for j in range(p):
        X = np.concatenate((X, y[(p-j):(T-h-j+1)]), axis = 1)
        
    # do OLS regression
    b = LA.inv(X.T.dot(X)).dot(X.T.dot(y[(p+h-1):]))
    yreg[(p+h-1):] = y[(p+h-1):] - X.dot(b)
    
    return yreg
