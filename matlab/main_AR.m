clear; close; clc;
addpath(genpath(pwd));

%% FORECASTING FRED-MD WITH SIMPLE AUTOREGRESSIVE MODEL
%  author: Victor Sellemi

%% READ IN AND PREPARE DATA

% read in fred-md stationary transformed series
data = readtable('fred_md_stationary.csv');

% set dates
date = datetime(data.sasdate);

% set of forecasting targets 
INDPRO = data.INDPRO;                   % industrial production
UNRATE = data.UNRATE;                   % unemployment rate
CPI    = data.CPIAUCSL;                 % inflation as captured by cpi
SPREAD = data.GS10 - data.FEDFUNDS;     % 10-year T rate - Fed funds rate
HOUST  = data.HOUST;                    % housing starts

% split data in-sample (training), hyper-parameter tuning (validation), and
% out-of-sample (test) -- (1/3, 1/3, 1/3) split
len = ceil(length(date) / 3);
train_ind = 1:len; 
val_ind   = (len+1):(2*len);
test_ind  = (2*len+1):length(date); 

%% SETTINGS

% pick the forecasting target
YY = CPI; 

% pick the forecasting horizon
h = 1; % 3,9,12,24

% pick the maximum number of lags
p_max = 12; 

% pick the in-sample information criterion for model selection: 
% AIC = 1, BIC = 2, HQ = 3
ICind = 2; 

% rolling window indicator for pseudo-oos (default is expanding)
roll = 0; 

% rolling window length (if roll=1);
wL = 36;

% specify benchmark for out-of-sample performance comparison (random walk
% (RW) or prevailing mean (PM)
benchmark = 'RW';

% max lag order in predictor
py_max     = 12;

% index all possible combinations of hyperparameters
combinations = reshape(ndgrid(1:py_max),[],1);


%% AR MODEL - IN-SAMPLE CRITERIA, PSEUDO-OOS FIT

% preallocate for validation (lag-length) selection 
insampIC = NaN(size(combinations,1),3);

% implement autoregressive model with in-sample lag length selection
for i = 1:size(combinations,1)
    
    p = combinations(i,1);
    
    % initialize target and predictors
    Y = YY(train_ind);
    X = lagmatrix(Y,0:p); X = X((p+1):(end-h), :); X = [ones(length(X),1), X];
    Y = Y((1+h+p):end);
    
    % estimate OLS
    S = OLS(X,Y);
    
    % evaluate on validation
    Y = YY(val_ind);
    X = lagmatrix(Y,0:p); X = X((p+1):(end-h), :); Xval = [ones(length(X),1), X];
    Yval = Y((1+h+p):end);
    
    yhat = Xval * S.Beta;
    
    % store results
    insampIC(p,:) = IC(Yval,yhat, length(Yval), size(Xval,2));    
    
end


%% POOS FORECAST

% pick best model
[~,best_i] = min(insampIC(:,ICind)); 
p = combinations(best_i,1);

% preallocate oos forecast
yHat = NaN(length(test_ind),1);   

if roll
     
     for t = 1:length(test_ind)
        
        % initialize data up to t in test set
        endInd = test_ind(t) - h - 1;

        Y = YY((endInd-wL-p-h):endInd);
        X = lagmatrix(Y,0:p); X = X((p+1):(end-h), :); X = [ones(length(X),1), X];
        Y = Y((1+h+p):end);
        
        % estimate OLS
        S = OLS(X,Y);
        
        % forecast for t+1
        Y = YY((endInd-p+1):(endInd + 1));
        X = [1, Y'];
        
        yHat(t) = X * S.Beta;
        
    end
    
else
       
     for t = 1:length(test_ind)
        
        % initialize data up to t in test set
        endInd = test_ind(t) - h- 1;

        Y = YY(1:endInd);
        X = lagmatrix(Y,0:p); X = X((p+1):(end-h), :); X = [ones(length(X),1), X];
        Y = Y((1+h+p):end);
        
        % estimate OLS
        S = OLS(X,Y);
        
        % forecast for t+1
         Y = YY((endInd-p+1):(endInd + 1));
        X = [1, Y'];
        
        yHat(t) = X * S.Beta;
        
    end
    
end

% true value
Y = YY(test_ind);

% plot forecast vs. true
plot_ts(date(test_ind), [Y, yHat], '', '',1, {'True', 'Forecast'}, 0)

% MSPE calculations
% 3 year window for rolling rmse
wL = 36;

switch benchmark
    case 'RW'
        ytemp      = forecast_RW(YY);
        bench_eval = MSPE(YY(test_ind), ytemp(test_ind),wL);
    case 'PM'
        ytemp      = forecast_PM(YY);
        bench_eval = MSPE(YY(test_ind), ytemp(test_ind),wL);
end


eval = MSPE(Y,yHat,wL);
disp(join(['Out-of-sample total MSPE is: ', num2str(eval.MSPE, "%.4f")]))

% Cumulative MSPE
plot_ts(date(test_ind),[eval.CUM_MSPE, bench_eval.CUM_MSPE], ...
    '', 'Cumulative RMSE',2, {'AR', benchmark}, 0)

% Rolling RMSPE
plot_ts(date(test_ind((1+wL):end)),[eval.ROLL_RMSPE, bench_eval.ROLL_RMSPE],...
    '', 'Rolling RMSE',3, {'AR', benchmark}, 0 )

%% OOS PERFORMANCE TESTS

% DM test
e1 = eval.errors;
e2 = bench_eval.errors;
[DM,pval_L,pval_LR,pval_R]= dmtest(e1, e2, h);

% MZ test
[MZstat, MZpval]=MZtest(Y,yHat);

% White and Hansen P-vals : c = consistent, u = upper, l = lower
[c,u,l]=bsds(e2,e1,1000,12,'STUDENTIZED','STATIONARY');

% model confidence set
[includedR,pvalsR,excludedR,includedSQ,pvalsSQ,excludedSQ]=mcs([e1,e2],0.05,1000,12,'STATIONARY');

%% SAVE OOS ERROR FOR LATER ANALYSIS
AR_errors = e1;
if isfile('FORECAST_ERRORS.mat'), save('FORECAST_ERRORS.mat', 'AR_errors','-append'); end





