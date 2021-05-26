#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:40:40 2021

@author: victorsellemi
"""

import numpy as np

def filter_MA(Y,q = 2):
    """
    DESCRIPTION: 
        Decompose a time series into a trend and stationary component 
        using the moving average (MA) filter (i.e., low pass filter)
        
    INPUT:
        Y = (T x 1) vector of time series data
        q = scalar value of moving average (half) window: default = 2

        
    OUTPUT: 
        trend = (T x 1) vector of trend component of the time series, i.e., low frequency component
        error = (T x 1) vector of stationary part of the time series
    

    """
    
    # length of time series
    T = Y.shape[0]
    
    # window width
    Q = 2*q
    
    # border of the series is preserved
    p1 = np.concatenate((np.eye(q), np.zeros((q,T-q))), axis = 1)
    p2 = np.zeros((T-Q,T))
    p3 = np.concatenate((np.zeros((q,T-q)), np.eye(q)), axis = 1)
    P  = np.concatenate((p1,p2,p3), axis = 0)
    
    # part of the series to be averaged
    X = np.eye(T-Q)
    Z = np.zeros((T-Q,1))
    
    for i in range(Q):
        
        # update X
        X = np.concatenate((X, np.zeros((T-Q,1))), axis = 1) + np.concatenate((Z, np.eye(T-Q)), axis = 1)
        
        # update Z
        Z = np.concatenate((Z, np.zeros((T-Q,1))), axis = 1)
        
    
    X = np.concatenate((np.zeros((q,T)), X, np.zeros((q,T))), axis = 0)
    
    # construct linear filter
    L = P + (1/(Q+1)) * X
    
    # construct the trend
    trend = L.dot(Y)
    
    # construct stationary component
    signal = Y - trend
    
    
    return trend,signal
