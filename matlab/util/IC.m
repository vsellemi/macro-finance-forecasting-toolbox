% Description: this function calculates information criteria from
% forecast
% Author: Victor Sellemi

% INPUT: 
%        - y = (T x 1) vector of true values
%        - yhat = (T x 1) vector of predictions
%        - nobs = number of observations
%        - nreg = number of predictors
% OUTPUT: 
%        - icvec = (1x3) vector of (aic,bic,hqic)

function icvec = IC(y,yhat,nobs,nreg)

% residuals
res=y-yhat; 

% explained sum of squared
ESS=res'*res;             

% information criteria
Aic=log((ESS/nobs))+(2*(nreg))/nobs;           %Akaike (minimize)
Bic=log((ESS/nobs))+((nreg)*log(nobs))/nobs;   %Bic    (minimize)
HQ=log((ESS/nobs))+2*nreg*log(log(nobs))/nobs; %HQ     (minimize)

icvec = [Aic, Bic, HQ]; 

end