#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 

@author: victorsellemi
"""

import numpy as np
from scipy.stats import norm       

def dmtest(e1, e2, h):
    """
    DESCRIPTION: 
        function to implement Diebold-Mariano test
    INPUT: 
        e1 = error of first prediction and actual values
        e2 = error of second prediction and actual values
        h  = forecast horizon
    OUTPUT: 
        DM = DM statistic
        pval_L = one sided test p_value for H1: MSE1 > MSE2
        pval_R = one sided test p_value for H1: MSE1 < MSE2
        pval_LR = two sided test p_value for H1: MSE1 != MSE2
    """
    
    T = e1.shape[0]
    
    # make sure error vectors have same dimensions
    if e1.shape != e2.shape:
        e1 = e2.reshape(e1.shape) 
        
    d = e1**2 - e2**2
    
    # calculate variance of loss differential, taking into account autocorrelation
    dMean = np.mean(d)
    gamma0 = np.var(d)
    if h > 1:
        gamma = np.zeros((h-1,1))
        for i in range(h-1):
            sampleCov = np.cov(d[i:T], d[:(T-i)])
            gamma[i] = sampleCov[0,1] 
        varD = gamma0 + 2*np.sum(gamma)
    else:
        varD = gamma0
    
    # compute Diebold-Mariano statistic DM~N(0,1)
    DM = dMean / np.sqrt((1/T)*varD) # equivalent to R = OLS(ones(T,1), d); R.tstat == DM
    
    # one sided test H0: MSE1 = MSE2, H1: MSE1 > MSE2
    pval_R = 1 - norm.cdf(DM, 0, 1)
    
    # one sided test H0: MSE1 = MSE2, H1: MSE1 < MSE2
    pval_L = 1 - norm.cdf(DM, 0 ,1)
    
    # two sided test
    if DM > 0: 
        pval_LR = (1 - norm.cdf(DM,0,1)) + norm.cdf(-DM, 0, 1)
    if DM <= 0 or np.isnan(DM):
        pval_LR = (1 - norm.cdf(-DM,0,1)) + norm.cdf(DM, 0, 1)
    
    return DM, pval_L, pval_LR, pval_R
