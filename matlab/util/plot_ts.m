% Description: this function plots time series 
% Author: Victor Sellemi

% INPUTS:
%        - date: Tx1 date vec
%        - series: Txk series
%        - labels: 1xk cell array of lables

function p = plot_ts(date, series, xlab, ylab, fignum, labels, standardize) 

figure(fignum);

if nargin < 6
    plot(date,series,'.-');
    grid on;
    xlabel(xlab);
    ylabel(ylab);
    set(gca, 'fontname', 'cmr12');
    
else
    
    if standardize
        series = (series - nanmean(series, 1)) ./ std(series,0,1,'omitnan');
    end
    
    p = plot(date, series, '.-');
    grid on;
    xlabel(xlab); ylabel(ylab);
    set(gca, 'fontname', 'cmr12');
    legend(labels);
    
end





end