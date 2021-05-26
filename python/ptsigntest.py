#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 

@author: victorsellemi
"""

import numpy as np
from scipy.stats import norm

def PTSignTest(y, yhat): 
    """
    DESCRIPTION: 
        This function implements the Pesaran-Timmermann Sign Test
    INPUT: 
        y = vector of observations (Tx1)
        y = vector of forecasts  (Txk) for k different forecasts
    OUTPUT:
        signPvals = list of k p_values corresponding to the one-sided test
    NOTES: 
        rejection of this test supports directional accuracy

    """
    
    TOut = y.shape[0]
    y = y.reshape((TOut,1))
    k = yhat.shape[1]
    
    directionBoth = y * yhat > 0
    directionF = yhat > 0 
    directionY = y > 0

    pHat = np.mean(directionBoth, axis = 0)
    pHatY = np.mean(directionY)
    pHatF = np.mean(directionF, axis = 0)

    signPvals = []

    for ii in range(k):
        pHatStar = pHatY * pHatF[ii] + (1-pHatY) * (1-pHatF[ii])
        varPHat = (1/TOut) * pHatStar * (1-pHatStar)
        varPStar = (1/TOut)*((2*pHatY-1)**2)*pHatF[ii]*(1-pHatF[ii]) + (1/TOut)*((2*pHatF[ii]-1)**2)*pHatY*(1-pHatY) + (4/(TOut**2))*pHatY*pHatF[ii]*(1-pHatY)*(1-pHatF[ii])
        signStat = (pHat[ii]-pHatStar)/np.sqrt(varPHat - varPStar)
        if np.isnan(signStat):
            if pHat[ii] > pHatStar:
                signStat = np.inf
            else:
                signStat - -np.inf
        signPvals.append(1- norm.cdf(signStat))

    return signPvals 
