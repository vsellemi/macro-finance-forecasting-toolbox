% Description: this function prepares data for forecasting regression by
% adding lags to predictor matrix

% Author: Victor Sellemi

% INPUT: 
%        - Y     = (T x 1) vector of forecasted variable
%        - F     = (T x k) matrix of predictors
%        - pf    = number of lags in F
%        - h     = forecasting horizon
%        - const = binary indicator to add constant as predictor
%        - py    = number of autoregressive lags to include
% OUTPUT: 
%        - X     = (T - h - max(py,pf) x nf*(pf+1) + py + 1 + const) new matrix of predictors
%        - Y     = (T - h - max(py,pf) x 1) new vector of dependent variable

function [X,Y] = add_lags(Y,F,pf,h,const,py)
    
    if nargin < 5
       
       F = lagmatrix(F,0:pf); 
       F = F((max(py,pf)+1):(end-h), :);
       X = F; 
       Y = Y((1+h+pf):end);
       
    else
        
        F = lagmatrix(F,0:pf); F = F((max(py,pf)+1):(end-h), :);
        X = lagmatrix(Y,0:py); X = X((max(py,pf)+1):(end-h), :);
        X = [X,F]; 
        Y = Y((1+h+max(py,pf)):end);
        
    end
    
    if const
        X = [ones(size(X,1),1), X]; 
    end



end