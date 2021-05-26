function [cdf_y,cdf_x] = ecdf_equally_spaced(y,n)

% PURPOSE: compute empirical distribution for density y, using N equal
% increments - size of increment is (max(y)-min(y))/(n-1) when n is a
% scalar, or n is the grid itself when not a scalar;

if isscalar(n) 
    cdf_x = linspace(mean(y)-10*std(y),mean(y)+10*std(y),n)';
else
    cdf_x = n;
end

cdf_y = NaN(length(cdf_x),1);

for i=1:length(cdf_x)
    cdf_y(i) = sum(y<=cdf_x(i))/length(y);
end

% error check
if max(cdf_y) == 0
    error('Actual y not included in x range considered');
end

    

