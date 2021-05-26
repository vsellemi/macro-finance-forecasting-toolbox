# Time Series Toolbox in Python
Python functions for time series data analysis

## Forecast Evaluation

'dmtest.py': implements Diebold-Mariano test

'mztest.py': implements Mincer-Zarnowitz test

'ptsigntest.py': implements Pesaran-Timmermann sign test

## Bootstrap

'block_bootstrap.py': implements circular block bootstrap

'stationary_bootstrap.py': implements stationary bootstrap

'mcs.py': calculates the model confidence set of Hansen, Lunde, Nason (2011) (requires 'block_bootstrap.py' and 'stationary_bootstrap.py')

## Filtering

'filter_MA.py' : moving average filter

'seasonal_adj.py' : seasonal adjustment as in Brockwell & Davis (1991)

'hamilton_filter/rwcyc' : extract cyclical component of time series for random walk baseline as in Hamilton (2018)

'hamilton_filter/regcyc' : extract cyclical component of time series from linear regression as in Hamilton (2018)
