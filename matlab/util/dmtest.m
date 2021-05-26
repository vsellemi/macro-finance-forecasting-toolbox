
function [DM pval_L pval_LR pval_R]= dmtest(e1, e2, h)

T = size(e1,1);
d = e1.^2 - e2.^2;
% Calculate the variance of the loss differential, taking into account autocorrelation
dMean = mean(d);
gamma0 = var(d);
if h > 1
    gamma = zeros(h-1,1);
    for i = 1:h-1
        sampleCov = cov(d(1+i:T),d(1:T-i));
        gamma(i) = sampleCov(2);
    end
    varD = gamma0 + 2*sum(gamma);
else
    varD = gamma0;
end

% Compute the diebold mariano statistic DM ~N(0,1)
DM = dMean / sqrt ( (1/T)*varD );  %equivalent to R=OLS(ones(T,1),d); R.tstat==DM  
    
%one sided test H0: MSE1=MSE2, H1=MSE1>MSE2
pval_R=1-normcdf(DM,0,1); 
%one sided test H0: MSE1=MSE2, H1=MSE1<MSE2
pval_L=normcdf(DM,0,1); 
%two side test H0: MSE1=MSE2, H1=MS1 different from MSE2
if DM>0;
pval_LR=(1-normcdf(DM,0,1))+normcdf(-DM,0,1); 
elseif DM<=0 || isnan(DM)
pval_LR=(1-normcdf(-DM,0,1))+normcdf(DM,0,1);     
end