% Description: this function calculates n-lag random walk forecast of a series
% Author: Victor Sellemi

% INPUT: 
%        - y    = (Tx1) vector of time series to forecast
%        - lag  = number of lags in the RW model 
% OUTPUT: 
%        - yhat = forecasted values (first entry is NaN)


function yhat = forecast_RW(y,lag)

if nargin < 2
    lag = 1;
end

yhat = NaN(size(y));
yhat(2:end) = movmean(y(1:(end-1)),[lag,0]);

end