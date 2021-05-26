#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as random

def stationary_bootstrap(data, B, w):
    '''
    Implements stationary bootstrap for stationary, dependent series
    
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
    
    # Define the probability of a new block
    p = 1 / w
    
    # Set up bsdata and indices
    indices = np.zeros((t,B))
    
    # Initial positions
    indices[0,:] = np.ceil(t*random.rand(1,B))
    
    # Set up the random numbers
    select = random.rand(t,B) < p
    indices[select] = np.ceil(random.rand(1,sum(sum(select))) * t).reshape(indices[select].shape)
    
    for i in np.arange(1,t,1):
        # Determine whether we stay (rand > p) or move to a new starting value (rand < p)
        indices[i, ~select[i,:]] = indices[i-1, ~select[i,:]] + 1
    
    indices[indices >= t] = indices[indices >= t] - t
    indices = indices.astype(int)
    
    bsdata = data[indices].squeeze()
    
    return bsdata, indices


