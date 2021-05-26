#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21

@author: victorsellemi
"""

import numpy as np
import numpy.linalg as LA
from scipy import optimize     
from scipy.stats import f        

def MZtest(y,yhat):
    """
    DESCRIPTION: 
        function to implement Mincer-Zarnowitz Test
    INPUT:
        y = true series to be forecasted
        yhat = forecasted series
    OUTPUT:
        MZStat = MZ test statistic
        MZpval = MZ test p_value    
    """
    MZstat = []
    MZpval = []
    for ii in range(yhat.shape[1]):
        intercept = np.ones((y.shape[0],1))
        X = np.concatenate((intercept, yhat[:,[ii]]), axis = 1)
        x,_, _, _= LA.lstsq(X,y, rcond = None)
        resnormU = np.sum((x[0]*X[:,[0]] + x[1]*X[:,[1]] - y)**2)
        SSE = np.sum(resnormU)
        N_restriction = 2
        cons = ({'type' : 'ineq', 'fun': lambda x: -x[0]},
                {'type': 'ineq', 'fun': lambda x: -x[1] + 1} )
        sol = optimize.minimize(lambda x: np.sum((x[0]*X[:,[0]] + x[1]*X[:,[1]] - y)**2), x0 = np.ones((2,1)), constraints = cons)
        #sol = optimize.minimize(lambda x: LA.norm(X.dot(x) - y), x0 = np.array([[1],[1]]), constraints = cons)
        x = sol.x
        resnormR = np.sum((x[0]*X[:,[0]] + x[1]*X[:,[1]] - y)**2)
        SSER = np.sum(resnormR)
        N = len(y)
        k = X.shape[1]
        df = N - k
        temp_MZstat = ((SSER - SSE) / N_restriction) / (SSE / df) 
        MZstat.append(temp_MZstat)
        MZpval.append( 1 - f.cdf(temp_MZstat, N_restriction, df))
    return np.array(MZstat), np.array(MZpval)
