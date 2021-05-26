clear; close; clc;
addpath(genpath(pwd));

%% FORECASTING FRED-MD WITH MS as in Pettenuzzo & Timmermann (2015)

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

% MS Hyperparameters
%    Parameters controlling estimation, including lag length choices
p_Y      = 4; %number of lags of dep variable (when present)
q_X      = 1; %number of lags of exogenous variable (when present)

%    Parameters controlling MS and CP estimation (dictating max number of
%       breaks/regimes tested)
K = 4;

%   MCMC parameters
I           = 1000;   % this is how many draws I want to keep
burn        = 100;    % this is how many draws I want to burn at the beginning of the chain
thin_factor = 1;     % this is how often I am going to retain draws

% priors
psi_mat     = 10;
v0_mat      = 0.01;

% Hyperparameters on MS transition prob matrix elements
ekk  = 2; % this is for the elements of the main diagonal
ekkp = 1; % this is for the off-diagonal elements

% Parameters pertaining to predictive density simulations
forc_rep    = 10;     % number of forecasts computed for each draw of Gibbs sampler

%% MS MODEL: IN-SAMPLE CRITERIA, PSEUDO-OOS FIT AND EVALUATION

% forecast with TVSVP (in-sample)
temp_ind = 1:(test_ind(end) - h);
[Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);

% set prior
prior = set_prior_ms(Xtr,Ytr,psi_mat,v0_mat,ekk,ekkp,K);

% estimate MS parameters with Gibbs sampler
gibbs = Gibbs_MS1(Ytr,Xtr,prior,K,I,burn,thin_factor);

% time-series of states
S    = round(mean(gibbs.S,2));

% coefficients by state
betavec = mean(gibbs.beta,1);
beta    = reshape(betavec,K,size(Xtr,2)); % K x npred
betamat = NaN(size(Xtr));
for tt = 1:size(betamat,1)
    betamat(tt,:) = beta(S(tt),:);
end

yhat = sum(betamat .* Xtr,2);


%% PLOT MS PROBS

fullscreen = get(0,'ScreenSize');
figure('Position',[0 0 fullscreen(3) fullscreen(4)]);
for j=1:K
    
    subplot(K,1,j)
    plot(date((max(py,pf)+1):(end-h-1)),mean(gibbs.Pr(:,:,j),2),'.-b','linewidth',1);
    ylim([0 1]);
    
    title(['Prob of regime # ',num2str(j)]);
    
    ylabel('Filtered prob')
    datetick('x','yyyy','keepticks','keeplimits')
    set(gca,'Xgrid','on','YGrid','on')
    set(gca,'FontSize',14);
    h_ylabel = get(gca,'YLabel');
    set(h_ylabel,'FontSize',14);
    h_title = get(gca,'Title');
    set(h_title,'FontSize',14);
end


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
        
        % estimate ms on rolling window
        [Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);
        
        % set prior
        prior = set_prior_ms(Xtr,Ytr,psi_mat,v0_mat,ekk,ekkp,K);
        
        % estimate MS parameters with Gibbs sampler
        gibbs = Gibbs_MS1(Ytr,Xtr,prior,K,I,burn,thin_factor);
        
        % states
        S    = round(mean(gibbs.S,2));
        
        % coefficients by state up to time t
        betavec = mean(gibbs.beta,1);
        beta    = reshape(betavec,K,size(Xtr,2)); % K x npred
        
        % transition probability matrix
        Pmat = squeeze(mean(gibbs.Pr,2));
        
        % forecast for t+h
        temp_ind = (endInd-wL-p-h):(endInd);
        [Xte,Yte] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);
        Xte = Xte(end,:);
        [~,S_next] = max(Pmat(S(end),:));
        beta_next = beta(S_next,:)'; 
        ythat = Xte * beta_next;
        
        yHat(t) = ythat;
        
    end
    
else
    
    for t = 1:length(test_ind)
        
        % initialize data up to t in test set
        endInd = test_ind(t) - h;
        temp_ind = 1:(endInd-1);
        
         % estimate ms on expanding window
        [Xtr,Ytr] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);
        
        % set prior
        prior = set_prior_ms(Xtr,Ytr,psi_mat,v0_mat,ekk,ekkp,K);
        
        % estimate MS parameters with Gibbs sampler
        gibbs = Gibbs_MS1(Ytr,Xtr,prior,K,I,burn,thin_factor);
        
        % states
        S    = round(mean(gibbs.S,2));
        
        % coefficients by state up to time t
        betavec = mean(gibbs.beta,1);
        beta    = reshape(betavec,K,size(Xtr,2)); % K x npred
        
        % transition probability matrix
        Pmat = squeeze(mean(gibbs.Pr,2));
        
        % forecast for t+h
        temp_ind = (endInd-wL-p-h):(endInd);
        [Xte,Yte] = add_lags(YY(temp_ind),factors(temp_ind,1:nf),pf,h,1,py);
        Xte = Xte(end,:);
        [~,S_next] = max(Pmat(S(end),:));
        beta_next = beta(S_next,:)'; 
        ythat = Xte * beta_next;
        
        yHat(t) = ythat;
    end
    
end


%% EVALUATE OOS PERFORMANCE

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
    '', 'Cumulative RMSE',2, {'MS', benchmark}, 0)

% Rolling RMSPE
plot_ts(date(test_ind((1+wL):end)),[eval.ROLL_RMSPE, bench_eval.ROLL_RMSPE],...
    '', 'Rolling RMSE',3, {'MS', benchmark}, 0 )

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
MS_errors = e1;
if isfile('FORECAST_ERRORS.mat'), save('FORECAST_ERRORS.mat', 'MS_errors','-append'); end

