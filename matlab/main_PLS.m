clear; close; clc;
addpath(genpath(pwd))

%% FORECASTING FRED-MD WITH PARTIAL LEAST SQUARES
%  author: Victor Sellemi

%% READ IN AND PREPARE DATA

% read in fred-md stationary transformed series
data = readtable('fred_md_stationary.csv');

% read in pca factors
factors = readtable('fred_md_pca_factors.csv');
factors = factors{:,2:end};

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

% model selection method
%     ictr    = in-sample information criteria on training sample
%     icval   = in-sample information criteria on validation sample
%     pooscv  = pseudo oos cross-validation with expanding window (takes longer)
%     kfcv    = K-fold cross-validation
model_selection = 'icval';

% pick the in-sample information criterion for in-sample model selection
%  (only matters when model selection is icval or ictr)
%     Akaike (AIC)       = 1
%     Bayesian (BIC)     = 2
%     Hannan-Quinn (HQ)  = 3
icidx = 2;

% pick K if model selection method is K-Fold (default is 5)
K     = 5;

% rolling window indicator for pseudo-oos (default is expanding)
roll = 0;

% rolling window length (if roll=1);
wL = 36;

% specify benchmark for out-of-sample performance comparison (random walk
% (RW) or prevailing mean (PM)
benchmark = 'RW';

% max lag order in predictor
py_max     = 6;

% max lag order in factors
pf_max     = 6;

% max number of principal components
nf_max     = size(factors,2);

% PLS hyperparameters
%     tol2 = tolerance in convergence of PLS algorithm
tol2       = 1e-10;

% index all possible combinations of hyperparameters
[a,b,c] = ndgrid(1:py_max,1:pf_max,nf_max:nf_max);
combinations = [reshape(a,[],1), reshape(b,[],1), reshape(c,[],1)];

disp(join(['Testing ', num2str(size(combinations,1), "%.0f"), ' combinations of hyperparameters.']))

%% PLS MODEL - OPTIMIZE OVER HYPERPARAMETERS

insampIC = NaN(size(combinations,1),3);
val_err  = NaN(size(combinations,1),1);

for i = 1:size(combinations,1)
    
    tic;
    
    % set hyperparameters
    py     = combinations(i,1);
    pf     = combinations(i,2);
    nf     = combinations(i,3);
    
    % model evaluation
    switch model_selection
        
        case 'icval'
            
            % estimate model on in-sample part
            [Xtr,Ytr] = add_lags(YY(train_ind),factors(train_ind,1:nf),pf,h,0,py);
            S_pls     = PLS(Xtr,Ytr,tol2);
            
            % evaluate the model on the validation set
            [Xval,Yval] = add_lags(YY(val_ind),factors(val_ind,1:nf),pf,h,0,py);
            yhat        = Xval * (S_pls.P * S_pls.B * S_pls.Q');
            
            % store information criterial
            insampIC(i,:) = IC(Yval,yhat, length(Yval), size(Xval,2));
            
        case 'ictr'
            
            % initialize target and predictors
            [Xtr,Ytr] = add_lags(YY(train_ind),factors(train_ind,1:nf),pf,h,0,py);
            S_pls     = PLS(Xtr,Ytr,tol2);
            
            % in-sample evaluation on training set
            yhat = Xtr * (S_pls.P * S_pls.B * S_pls.Q');
            insampIC(i,:) = IC(Ytr,yhat, length(Ytr), size(Xtr,2));
            
        case 'pooscv'
            
            % pseudo-oos evaluation (using last 25% of in-sample data)
            yhat = NaN(length(val_ind), 1);
            
            for t = 1:length(val_ind)
                
                % initialize data up to t in test set
                endInd = val_ind(t) - h;
                temp_ind = 1:(endInd-1);
                
                [Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,0,py);
                S_pls     = PLS(Xtr,Ytr,tol2);
                
                % forecast for t+h
                temp_ind = 1:(endInd);
                [Xte,Yte] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,0,py);
                Xte = Xte(end,:);
                ythat = Xte * (S_pls.P * S_pls.B * S_pls.Q');
                
                yhat(t) = ythat;
            end
            
            ytrue = YY(val_ind);
            eval_cv = MSPE(ytrue,yhat);
            val_err(i) = eval_cv.MSPE;
            
        case 'kfcv'
            
            % default is 5-Fold CV
            try K; catch; K=5; end
            split = floor(linspace(1,val_ind(end),K+2));
            
            tmp_errs = NaN(K,1);
            
            for k = 1:K
                
                cvidxtr = split(1):split(k+1);
                cvidxte = split(k+1):split(k+2);
                
                % train up to fold k
                [Xtr,Ytr] = add_lags(YY(cvidxtr),factors(cvidxtr,1:nf),pf,h,0,py);
                S_pls     = PLS(Xtr,Ytr,tol2);
                
                % evaluate on fold k+1
                [Xte,Yte] = add_lags(YY(cvidxte),factors(cvidxte,1:nf),pf,h,0,py);
                yhat =  Xte * (S_pls.P * S_pls.B * S_pls.Q');
                
                eval_cv = MSPE(yhat,Yte);
                tmp_errs(K) = eval_cv.MSPE;
                
            end
            
            val_err(i) = mean(tmp_errs);
            
    end
    
    % estimate time remaining
    elapsed   = toc;
    remaining = ceil(elapsed * (size(combinations,1) - i) / 60);
    
    % progress updates
    if mod(i,100) == 0
        disp(join([i, "/", size(combinations,1)]))
        disp(join(["Approximately" remaining, "minutes remaining."]))
    end
    
end

% pick best model
switch model_selection
    case 'ictr'
        [~,best_i] = min(insampIC(:,icidx));
    case 'icval'
        [~,best_i] = min(insampIC(:,icidx));
    case 'pooscv'
        [~,best_i] = min(val_err);
    case 'kfcv'
        [~,best_i] = min(val_err);
end

%% PLS MODEL: IN-SAMPLE CRITERIA, PSEUDO-OOS FIT AND EVALUATION

py     = combinations(best_i,1);
pf     = combinations(best_i,2);
nf     = combinations(best_i,3);
p      = max(pf,py);

% preallocate oos forecast
yHat = NaN(length(test_ind),1);

if roll
    
    for t = 1:length(test_ind)
        
        % initialize data up to t in test set
        endInd = test_ind(t) - h;
        temp_ind = (endInd-wL-p-h):(endInd-1);
        
        % estimate PLS on rolling window
        [Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,0,py);
        S_pls     = PLS(Xtr,Ytr,tol2);
        
        % forecast for t+h
        temp_ind = (endInd-wL-p-h):(endInd);
        [Xte,Yte] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,0,py);
        Xte = Xte(end,:);
        ythat = Xte * (S_pls.P * S_pls.B * S_pls.Q');
        
        yHat(t) = ythat;
        
    end
    
else
    
    for t = 1:length(test_ind)
        
        % initialize data up to t in test set
        endInd = test_ind(t) - h;
        temp_ind = 1:(endInd-1);
        
        % estimate PLS on expanding window
        [Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,0,py);
        S_pls     = PLS(Xtr,Ytr,tol2);
        
        % forecast for t+h
        temp_ind = 1:(endInd);
        [Xte,Yte] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,0,py);
        Xte = Xte(end,:);
        ythat = Xte * (S_pls.P * S_pls.B * S_pls.Q');
        
        yHat(t) = ythat;
        
    end
    
end


%% EVALUATE OOS PERFORMANCE

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
    '', 'Cumulative RMSE',2, {'PLS', benchmark}, 0)

% Rolling RMSPE
plot_ts(date(test_ind((1+wL):end)),[eval.ROLL_RMSPE, bench_eval.ROLL_RMSPE],...
    '', 'Rolling RMSE',3, {'PLS', benchmark}, 0 )

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
PLS_errors = e1;
if isfile('FORECAST_ERRORS.mat'), save('FORECAST_ERRORS.mat', 'PLS_errors','-append'); end





