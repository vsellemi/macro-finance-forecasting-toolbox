% Description: this function implements OLS regression
% Author: Victor Sellemi

% INPUT: 
%        - X = (n x k) matrix of regressors
%        - y = (n x 1) vector of dependent variables
% OUTPUT: 
%        - S = MATLAB structure containing output items

function [S]=OLS(X,y)

[nobs, nreg]=size(X);

% coefficient
Beta=X\y;

% fitted y
yhat=X*Beta;

% residuals
res=y-yhat;  

% variance of residuals
shat=(res'*res)/(nobs-nreg);           

% variance-covariance matrix
varcovar=shat * (eye(nreg) / (X'*X));  

% tstat
tstat=Beta./sqrt(diag(varcovar));

% explained sum of squared
ESS=res'*res;             

% total sum of squares
TSS=(y-mean(y))'*(y-mean(y)); 

% R^2
Rq=1-(ESS/TSS);                        

% adjusted R^2 (maximize)
stilda=ESS/(nobs-(nreg-1)-1);
S2=(TSS)/(nobs-1);
Rqad=1-(stilda/S2);     

% information criteria
Aic=log((ESS/nobs))+(2*(nreg))/nobs;           %Akaike (minimize)
Bic=log((ESS/nobs))+((nreg)*log(nobs))/nobs;   %Bic    (minimize)
HQ=log((ESS/nobs))+2*nreg*log(log(nobs))/nobs; %HQ     (minimize)

S=struct('Beta',Beta,'yhat',yhat,'tstat',tstat,'res',res,'varRes', ...
    shat,'varcovarBeta',varcovar,'Rq',Rq,'Rqadj',Rqad,...
    'Akaike',Aic,'Bic',Bic,'HQ',HQ);

end