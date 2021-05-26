% Description: this function calculates prevailing mean forecast of a series
% Author: Victor Sellemi

% INPUT: 
%        - y    = (Tx1) vector of time series to forecast
% OUTPUT: 
%        - yhat = forecasted values (first entry is NaN)


function yhat = forecast_PM(y)

T = size(y,1);
yhat = NaN(T,1); 
yhat(2:end) = cumsum(y(1:(end-1))) ./ (1:(T-1))';

end