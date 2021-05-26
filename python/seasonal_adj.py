#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:29:49 2021

@author: victorsellemi
"""

import numpy as np
import filter_MA

def seasonal_adj(Y, quarter, q = 2):
    """
    DESCRIPTION:
        This function deseasonalizes data using method 2 from Brockwell & Davis (1991)
        
    INPUT: 
        - Y is a (T x 1) time-series vector to deseasonalize
        - quarter is a (T x 1) vector of seasonal period dummies taking values 1,...,2*q
        - q is a scalar which represents the moving average half window
        
    OUTPUT:
        - Y_deseasonalized is a (T x 1) time-series vector of deseasonalized data which needs
        to be detrended
        
    NOTES: default q=2 for quarterly data, required filter_MA.py to run

    """
    
    # Step 1) Detrend using a moving average filter whose period coincides 
    #         with one of the seasonal components e.g., for quarterly data
    #         the period component is 4 and moving average window will be set 
    #         to q = 2
    _,detrended = filter_MA(Y,q)
    
    # use Brockwell & Davis (1991) notation
    w = detrended
    
    # construct a matrix of detrended values for each quarter
    w = np.concatenate((np.expand_dims(w * (quarter == 1), axis = 1), 
                        np.expand_dims(w * (quarter == 2), axis = 1),
                        np.expand_dims(w * (quarter == 3), axis = 1),
                        np.expand_dims(w * (quarter == 4), axis = 1)), axis = 1)
    
    # calculate the average of detrended values of Y by quarter: obtain a 
    # (4 x 1) vector of average deviations from the trend by quarter
    w = np.array([[np.sum(w[:,0]) / np.sum(w[:,0]!=0)],
                  [np.sum(w[:,1]) / np.sum(w[:,1]!=0)],
                  [np.sum(w[:,2]) / np.sum(w[:,2]!=0)],
                  [np.sum(w[:,3]) / np.sum(w[:,3]!=0)]])
    
    # Since these average deviations do not necessarily sum to zero, we
    # estimate the seasonal component by tweaking them in such a way they
    # necessairly sum to zero:
    s = w - np.mean(w);
    
    # construct the (T x 1) vector of seasonal components
    T = Y.shape[0]
    d = np.zeros((T,1))
    d[quarter == 1] = s[0]
    d[quarter == 2] = s[1]
    d[quarter == 3] = s[2]
    d[quarter == 4] = s[3]
    
    # calculate deseasonalized data
    Y_deseasonalized = Y - d.reshape(Y.shape)
    
    return Y_deseasonalized
    
    
    
    
    
    
    