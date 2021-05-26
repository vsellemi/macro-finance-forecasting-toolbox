#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as random

def block_bootstrap(data, B, w):
    '''
    Implements circular block bootstrap for stationary, dependent series
    
    INPUTS: 
        data = T x 1 vector of data to be bootstrapped
        B    = number of bootstraps
        W    = block length
    
    OUTPUTS:
        bsdata  = T x B matrix of bootstrapped data
        indices = T x B matrix of locations of the original BSDATA = data[indexes]
    
    COMMENTS:
        To generate bootstrap sequences for other uses, such as bootstrapping vector
        processes, set data to (1:N)^T
    
    -Translated from Kevin Sheppard MFE MATLAB toolbox. 
    '''
    # ====================================================================== #
    # Input Checking    
    t,k = data.shape
    if k > 1:
        raise ValueError('DATA must be a column vector')
        
    if t < 2:
        raise ValueError('DATA must have at least 2 observations')
        
    if not np.isscalar(w) or w < 1 or np.floor(w) != w or w > t:
        raise ValueError('W must be a positive scalar integer smaller than T')
        
    if not np.isscalar(B) or B < 1 or np.floor(B) != B:
        raise ValueError('B must be a positive scalar integer')
    # ====================================================================== #
    
    # Compute the number of blocks needed
    s = int(np.ceil(t / w))
    
    # Generate the starting points
    Bs = np.floor(random.rand(s,B)*t) + 1
    indices = np.zeros((s*w, B))
    index = 0
    
    # Adder is a variable that needs to be added each loop
    adder = np.kron(np.ones((1,B)), np.arange(0,w,1).reshape((w,1)))
    for i in np.arange(0,t+1,w):
        indices[i:(i+w),:] = np.kron(np.ones((w,1)),Bs[index,:].reshape((1,B))) + adder
        index += 1
    
    indices = indices[:t,:]
    indices[indices >= t] = (indices[indices >= t] - t)
    indices = indices.astype(int)
    
    bsdata = data[indices].squeeze()
    
    return bsdata, indices


