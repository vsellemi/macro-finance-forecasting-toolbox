% Description: this function calculates the mean squared prediction error
% of a forecast
% Author: Victor Sellemi

% INPUT: 
%        - Y = (Tx1) of true values
%        - Yhat = (Tx1) vector of forecasts
% OUTPUT: 
%        - S = MATLAB structure containing output items

function [S] = MSPE(Y,Yhat,wL)

% default 3 years window length for rolling rmse (monthly)
if nargin < 3, wL = 36; end

T = length(Y); 
mspe_vec = (Y - Yhat).^2;
mspe_tot = sum(mspe_vec); 
roll_rmspe = zeros(T - wL,1); 
cum_mspe = cumsum(mspe_vec); 
for t = 1:(T-wL)
    roll_rmspe(t) = sqrt(sum(mspe_vec(t:(t+wL))));
end

S=struct('MSPE',mspe_tot,'ROLL_RMSPE',roll_rmspe, ...
    'CUM_MSPE', cum_mspe, 'errors', mspe_vec);

end