clear; close; clc;
addpath(genpath(pwd));

%% FORECASTING FRED-MD WITH TVPSV as in Pettenuzzo & Timmermann (2015)

%% READ IN AND PREPARE DATA

% read in fred-md stationary transformed series
data = readtable('fred_md_stationary.csv');

% read in pca factors
factors = readtable('fred_md_pca_factors.csv');
factors = factors{:,2:end};

% factors = data.UNRATE * 100;
% t = 1:numel(factors); nanx = isnan(factors);
% factors(nanx) = interp1(t(~nanx), factors(~nanx), t(nanx));

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
YY = CPI*100;

% pick the forecasting horizon
h = 1; % 3,9,12,24

% specify benchmark for out-of-sample performance comparison (random walk
% (RW) or prevailing mean (PM)
benchmark  = 'RW';

% rolling window indicator for pseudo-oos (default is expanding)
roll = 0;

% rolling window length (if roll=1);
wL    = 36;

% lag order in predictor
py = 1;

% lag order in factors
pf = 1;

% number of principal components
nf = 1;

% TVPSV model hyperparameters
%          MCMC parameters
I           = 500;   % this is how many draws I want to keep
burn        = 100;    % this is how many draws I want to burn at the beginning of the chain
thin_factor = 1;      % this is how often I am going to retain draws

%          Prior hyperparameters
%          Hyperparameters on linear regression coefficients and volatility (when constant)
%          **Note: Pettenuzzo and Timmermann (2015) set psi=10,v0=.005 for infl and
%                   psi=25,v0=.005 for gdp growth
psi_mat = 10;   % tightness of the prior of coefficients
v0_mat  = 0.01; % informativeness of the prior

% Hyperparameters on TVP and SV laws of motions
k_Q         = 0.1;
k_xsi       = 0.1;
v_Q         = 10;
k_h         = 0.1;
v_xsi       = 10;

% Hyperparameters on AR-SV parameters

% Modified from Clark and Ravazzolo
LAM_mean = [0;0.9];
LAM_var  = diag([0.5^2;0.01^2]);

% Hyperparameters on AR-TVP parameters
GAM_mean = 0.8;
GAM_var  = 0.001^2;


%% TVPSV MODEL: IN-SAMPLE CRITERIA, PSEUDO-OOS FIT AND EVALUATION

% forecast with TVPSV (in-sample)
temp_ind = 1:(test_ind(end) - h);
[Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);

% set prior
prior = set_prior_tvpsv(Xtr, Ytr, LAM_mean, LAM_var, k_xsi, v_xsi, k_h, ...
    GAM_mean, GAM_var, psi_mat, v0_mat, k_Q, v_Q);

% estimate parameters with Gibbs sampler
gibbs = Gibbs_AR_TVPSV1(Ytr,Xtr,prior,I,burn,thin_factor);

% produce forecast
beta = mean(gibbs.beta,1);
beta_t = squeeze(mean(gibbs.beta_t,2));
beta_sum = beta + beta_t;
yhat = sum(beta_sum .* Xtr,2);

%% PLOT TVP

K=size(gibbs.beta,2);

fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);

for i=1:K
    subplot(ceil(K/2),2,i);
    plot(date((max(py,pf)+1):(end-h-1)),repmat(mean(gibbs.beta(:,i)),length(gibbs.y),1)+mean(gibbs.beta_t(:,:,i),2),'.-b','linewidth',1);
    
    set(gca,'XLim',get(gca,'XLim'),'Layer','top')
    
    title([['\beta_{',num2str(i-1),'}']]);
    
    set(gca,'Xgrid','on','YGrid','on')
    set(gca,'FontSize',10);
    h_ylabel = get(gca,'YLabel');
    set(h_ylabel,'FontSize',10);
    h_title = get(gca,'Title');
    set(h_title,'FontSize',10);
end

%% PLOT SV

fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
plot(date((max(py,pf)+1):(end-h-1)),mean(exp(gibbs.h_t),2),'.-b','linewidth',2);

set(gca,'XLim',get(gca,'XLim'),'Layer','top')

title(['SV']);
set(gca,'Xgrid','on','YGrid','on')
set(gca,'FontSize',14);
h_ylabel = get(gca,'YLabel');
set(h_ylabel,'FontSize',14);
h_title = get(gca,'Title');
set(h_title,'FontSize',14);

%% PSEUDO-OOS FORECASTING

% shortened sample since MCMC takes some time
test_ind = test_ind(1:72);

% preallocate oos forecast
yHat = NaN(length(test_ind),1);
p = max(py,pf);

if roll
    
    for t = 1:length(test_ind)
        
        % initialize data up to t in test set
        endInd = test_ind(t) - h;
        temp_ind = (endInd-wL-p-h):(endInd-1);
        
        % estimate tvpsv on rolling window
        [Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);
        
        % set prior
        prior = set_prior_tvpsv(Xtr, Ytr, LAM_mean, LAM_var, k_xsi, v_xsi, k_h, ...
            GAM_mean, GAM_var, psi_mat, v0_mat, k_Q, v_Q);
        
        
        % estimate parameters with Gibbs sampler
        gibbs = Gibbs_AR_TVPSV1(Ytr,Xtr,prior,I,burn,thin_factor);
        
        % coefficients up to time t
        beta = mean(gibbs.beta,1);
        beta_t = squeeze(mean(gibbs.beta_t,2));
        beta_sum = beta + beta_t;
        
        % forecast for t+h
        temp_ind = (endInd-wL-p-h):(endInd);
        [Xte,Yte] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);
        Xte = Xte(end,:);
        gamma = mean(gibbs.gam,1);
        beta_next = gamma'.*beta_sum(end,:)';
        ythat = Xte * beta_next;
        
        yHat(t) = ythat;
        
    end
    
else
    
    for t = 1:length(test_ind)
        
        % initialize data up to t in test set
        endInd = test_ind(t) - h;
        temp_ind = 1:(endInd-1);
        
        % estimate tvpsv on expanding window
        [Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);
        
        % set prior
        prior = set_prior_tvpsv(Xtr, Ytr, LAM_mean, LAM_var, k_xsi, v_xsi, k_h, ...
            GAM_mean, GAM_var, psi_mat, v0_mat, k_Q, v_Q);
        
        
        % estimate parameters with Gibbs sampler
        gibbs = Gibbs_AR_TVPSV1(Ytr,Xtr,prior,I,burn,thin_factor);
        
        % forecast for t+h
        temp_ind = 1:(endInd);
        [Xte,Yte] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);
        Xte = Xte(end,:);
        gamma = mean(gibbs.gam,1);
        beta_next = gamma'.*beta_sum(end,:)';
        ythat = Xte * beta_next;
        
        yHat(t) = ythat;
        
    end
    
end


%% EVALUATE OOS PERFORMANCE

Y = YY(test_ind);

% plot forecast vs. true
plot_ts(date(test_ind), [Y, yHat], '', '',3, {'True', 'Forecast'}, 0)

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
    '', 'Cumulative RMSE',4, {'TVSVP', benchmark}, 0)

% Rolling RMSPE
plot_ts(date(test_ind((1+wL):end)),[eval.ROLL_RMSPE, bench_eval.ROLL_RMSPE],...
    '', 'Rolling RMSE',5, {'TVSVP', benchmark}, 0 )

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
TVSVP_errors = e1;
if isfile('FORECAST_ERRORS.mat'), save('FORECAST_ERRORS.mat', 'TVSVP_errors','-append'); end

