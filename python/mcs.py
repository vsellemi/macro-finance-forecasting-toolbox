#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as random

def mcs(losses, alpha, B, w, boot = 'STATIONARY'):
    '''
    Compute the model confidence set of Hansen, Lunde, Nason

    INPUTS: 
        LOSSES = T x K matrix of losses
        ALPHA  = The final pval to use the MCS
        B      = Number of bootstrap replications
        W      = Desired block length
        BOOT   = 'STATIONARY' or 'BLOCK', Stationary is default
        
    OUTPUTS:
        INCLUDEDR  = included models using the R method
        PVALSR     = Pvals using R method
        EXCLUDEDR  = Excluded models using the R method
        INCLUDEDSQ = Included models using SQ method
        PVALSSQ    = Pvals using SQ method
        EXCLUDEDSQ = Excluded models using SQ method
   
    COMMENTS:
        This version of the MCS operates on quantities that shoudl be called 
        bads, such as losses. If the quantities of interest are goods, such as returns,
        simply call MCS with -1*LOSSES
    
    - Translated to Python from Kevin Sheppard's MFE MATLAB toolbox

    '''
    # ===================================================================== #
    # ========================= SUBFUNCTIONS ============================== #
    
    def block_bootstrap(data, B, w):
        '''
        Implements circular block bootstrap for stationary, dependent series 
        '''
        t,k = data.shape
        if k > 1:
            raise ValueError('DATA must be a column vector')
        
        if t < 2:
            raise ValueError('DATA must have at least 2 observations')
        
        if not np.isscalar(w) or w < 1 or np.floor(w) != w or w > t:
            raise ValueError('W must be a positive scalar integer smaller than T')
        
        if not np.isscalar(B) or B < 1 or np.floor(B) != B:
            raise ValueError('B must be a positive scalar integer')
    
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
    
    def stationary_bootstrap(data, B, w):
        '''
        Implements stationary bootstrap for stationary, dependent series
        '''
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
   
    # ===================================================================== #
    # ====================== INPUT CHECKING =============================== #
    t = losses.shape[0]
    
    if t < 2:
        raise ValueError('LOSSES must have at least 2 observations')
    
    if not np.isscalar(alpha) or alpha >= 1 or alpha <= 0:
        raise ValueError('ALPHA must be a scalar between 0 and 1')
    
    if not np.isscalar(B) or B < 1 or np.floor(B) != B:
        raise ValueError('B must be a positive scalar integer')
    
    if not np.isscalar(w) or w < 1 or np.floor(w) != w:
        raise ValueError('W must be a positive scalar integer')
    
    boot = str.upper(boot)
    
    if boot != 'STATIONARY' and boot != 'BLOCK':
        raise ValueError('BOOT must be either STATIONARY or BLOCK')
    
    # ===================================================================== #
    
    # 1. Compute the indices to use throughout the procedure
    if boot == 'BLOCK':
        bsdata = block_bootstrap(np.arange(1,t,1).reshape(t-1,1), B, w)[0]
    else: 
        bsdata = stationary_bootstrap(np.arange(1,t,1).reshape(t-1,1), B, w)[0]
        
    # All of these values can be computed once
    M0 = losses.shape[1]
    
    # The i,j element contains the l(i,t) - l(j,t)
    dijbar = np.zeros((M0,M0))
    for j in range(M0):
        dijbar[j,:] = np.mean(losses - np.kron(np.ones((1,M0)), losses[:,[j]]), axis = 0)
        
    # for each j, compute dijbar using the BSdata then compute the var(dijbar)
        
    # 2(a). 
    dijbarstar = np.zeros((M0,M0,B))
    for b in range(B):
        meanworkdata = np.mean(losses[bsdata[:,b], :], axis = 0)
        for j in range(M0):
            # the i,j element contains the l(b,i,t) - l(b,j,t)
            dijbarstar[j,:,b] = meanworkdata - meanworkdata[j]
    
    vardijbar = np.mean((dijbarstar - np.repeat(dijbar[:,:,np.newaxis],B,axis=2))**2, axis = 2)
    vardijbar = vardijbar + np.diag(np.ones((M0,1)))
    
    # 2(b).
    z0 = (dijbarstar - np.repeat(dijbar[:,:,np.newaxis],B,axis=2)) / np.sqrt(np.repeat(vardijbar[:,:,np.newaxis],B,axis=2))
    zdata0 = dijbar / np.sqrt(vardijbar)
    
    # Only these depend the set of selected models
    excludedR = np.nan * np.ones((M0,1))
    pvalsR = np.ones((M0,1))
    for i in range(M0-1): 
        included = np.setdiff1d(np.arange(0,M0,1), excludedR)
        m = len(included)
        z = z0[included,:,:][:,included,:]
        
        # max over the abs value of z in each matrix
        empdistTR = np.max(np.max(abs(z),axis = 0),axis=1)
        zdata = zdata0[included,:][:,included]
        TR = np.max(np.max(zdata))
        pvalsR[i] = np.mean(empdistTR[empdistTR > TR])
        
        # finally compute the model to remove, which depends on the max standardized average
        # 1. compute dibar
        dibar = np.mean(dijbar[included,:][:,included],axis = 0) * (m / (m-1))
        # 2. compute var(dibar)
        dibstar = np.mean(dijbarstar[included,:,:][:,included,:],axis = 0) * (m / (m-1))
        vardi = np.mean((dibstar.T - np.kron(np.ones((B,1)), dibar))**2, axis = 0)
        t = dibar / np.sqrt(vardi)
        
        # remove the max of t
        modeltoremove = np.argmax(t)
        excludedR[i] = included[modeltoremove]
    
    # The MCS pval is the max up to that point
    maxpval = pvalsR[0]
    for i in np.arange(1,M0,1):
        if pvalsR[i] < maxpval:
            pvalsR[i] = maxpval
        else:
            maxpval = pvalsR[i]
    
    # Add the final remaining model to excluded
    excludedR[-1] = np.setdiff1d(np.arange(0,M0,1), excludedR)
    
    # the included models are all of these where the first pval is > alpha
    pl = np.where(pvalsR < alpha)[0][0] 
    includedR = excludedR[pl:]
    excludedR = excludedR[:(pl-1)]
    
    excludedSQ = np.nan * np.ones((M0,1))
    pvalsSQ = np.ones((M0,1))
    for i in range(M0-1):
        included = np.setdiff1d(np.arange(0,M0,1), excludedSQ)
        m = len(included)
        z = z0[included,:,:][:,included,:]
        
        # max over the abs value of z in each matrix
        empdistTSQ = np.sum(np.sum(z**2, axis = 0)/2, axis =0 )
        zdata = zdata0[included,:][:,included]
        TSQ = np.sum(np.sum(zdata**2, axis = 0)/2, axis =0 )
        pvalsSQ[i] = np.mean(empdistTSQ[empdistTSQ > TSQ])
        
        # finally compute the model to remove, which depends on the max standardized average
        # 1. compute dibar
        dibar = np.mean(dijbar[included,:][:,included],axis = 0) * (m / (m-1))
        # 2. compute var(dibar)
        dibstar = np.mean(dijbarstar[included,:,:][:,included,:],axis = 0) * (m / (m-1))
        vardi = np.mean((dibstar.T - np.kron(np.ones((B,1)), dibar))**2, axis = 0)
        t = dibar / np.sqrt(vardi)
        
        # remove the max of t
        modeltoremove = np.argmax(t)
        excludedSQ[i] = included[modeltoremove]
    
    # The MCS pval is the max up to that point
    maxpval = pvalsSQ[0]
    for i in np.arange(1,M0,1):
        if pvalsSQ[i] < maxpval:
            pvalsSQ[i] = maxpval
        else:
            maxpval = pvalsSQ[i]
            
    # add the final remaining model ot be excluded
             
    # Add the final remaining model to excluded
    excludedSQ[-1] = np.setdiff1d(np.arange(0,M0,1), excludedSQ)
    
    # the included models are all of these where the first pval is > alpha
    pl = np.where(pvalsSQ < alpha)[0][0] 
    includedSQ = excludedR[pl:]
    excludedSQ = excludedR[:(pl-1)]
    
    return includedR,pvalsR,excludedR,includedSQ,pvalsSQ,excludedSQ

