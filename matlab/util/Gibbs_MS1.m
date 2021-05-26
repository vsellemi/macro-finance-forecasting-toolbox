function result=Gibbs_MS(EstY,EstX,prior_struct,M,I,burn,thin_factor)

% PURPOSE: This program uses the hidden markov chain of Chib 1998 to compute the
%         unknown regime switch for the normal inverted gamma case
% *************************************************************************
% USAGE: result=Gibbs_MS(EstY,EstX,EstLagY,prior_struct,M,I,burn,thin_factor)
% *************************************************************************

%% Pull out priors from prior structure
field_list = fieldnames(prior_struct);

for i=1:length(field_list)
    eval([char(field_list(i)),' = getfield(prior_struct,''',char(field_list(i)),''');']);
end

%% Prepare X matrix for regression
[T,~] = size(EstY);

% Combine lagged Y and lagged exogenous regressors
X     = EstX;
[~,K] = size(X);

%store all draws in the following matrices
Beta_draws       = NaN(K*M,(I-burn)/thin_factor);
sigma2_draws     = NaN((I - burn)/thin_factor,M);
P_draws          = NaN(M^2,(I-burn)/thin_factor);
Pr_draws         = NaN(T,(I-burn)/thin_factor,M);
S_draws          = NaN(T,(I-burn)/thin_factor);

% Calculate a few quantities outside the loop for later use
capv0inv = V0\eye(size(V0,1));
v0s02    = v0*s02;
indexSwitch=0;

% Draw from priors to get things started
% Draw transition probability matrix from its prior
[Pdraw,p,tm]=MSLinRegDrawP(T,M,zeros(T,1),Pmat);

% Draw states from transition prob matrix (for intial obs, use steady-state
% probs. For other obs, draw from p(s_t|s_t-1) distribution
pv=p;
sdraw = NaN(T,1);
for t=1:T
    sdraw(t,1)=draw_multinom(pv,1,M,1); % this is the same as sdraw(t,1)=find(mnrnd(1,pv,1)==1);
    pv=Pdraw(sdraw(t,1),:)'; % this select the row of the transition prob matrix based on which state the chain is in at time t
end

% Draw beta and sig2 from priors
betadraw   = NaN(K,M);
sigma2draw = NaN(1,M);

for i=1:M
    %draw from beta prior
    betadraw(:,i) = b0 + chol(V0)'*randn(K,1);
    
    %draw from h prior
    hdraw           = gamm_rnd(1,1,.5*v0,.5*v0s02);
    sigma2draw(1,i) = 1/hdraw;
end

%% Start Gibbs sampler
for irep=2:I
    
    % Print iterations
    if mod(irep,100) == 0
        if irep == 100
            fprintf('\r Iterations: %d \t',irep)
        else
            fprintf('%d \t',irep);
        end
    end
    
    %------------------------------------------------------------------------------------------
    %   STEP I: Draw betas and sigma2s
    %------------------------------------------------------------------------------------------
    betadraw   = NaN(K,M);
    
    %For each regime draw from the posterior distr of its parameters
    for i=1:M
        this_y = EstY(sdraw==i);
        this_X = X(sdraw==i,:);
        
        %draw from beta conditional on h
        capv1inv = capv0inv+ (1/sigma2draw(1,i))*(this_X'*this_X);
        capv1=inv(capv1inv);
        b1 = capv1*(capv0inv*b0 + (1/sigma2draw(1,i))*this_X'*this_y);
        bdraw=b1 + chol(capv1)'*randn(K,1);
        
        %draw from h conditional on beta
        v1 = v0 + length(this_y);
        s12 = ((this_y-this_X*bdraw)'*(this_y-this_X*bdraw)+v0s02)/v1;
        hdraw=gamm_rnd(1,1,.5*v1,.5*v1*s12);
        
        betadraw(:,i) = bdraw;
        sigma2draw(1,i) = 1/hdraw;
    end
    
    %------------------------------------------------------------------------------------------
    %   STEP II: Draw states
    %------------------------------------------------------------------------------------------
    % Compute p(y_t|s_t=i,Y^{t-1},betas,sigma2s,P)
    lnpdat = NaN(T,M);
    for i=1:M
        lnpdat(:,i) = log(normpdf(EstY,X*betadraw(:,i),repmat(sqrt(sigma2draw(1,i)),T,1)));
    end
    
    [~,filprobdraw,sdraw] = filterMarkovMixture(p,Pdraw,lnpdat,1);
    %------------------------------------------------------------------------------------------
    %   STEP III: Draw transition prob matrix
    %------------------------------------------------------------------------------------------
    [Pdraw,p,tm]=MSLinRegDrawP(T,M,sdraw,Pmat);
    
    %------------------------------------------------------------------------------------------
    %   Rearrange states according to constraint on sigma2 draws, where
    %  (sigma2(1) < (sigma2(2) < ... < (sigma2(M)
    %------------------------------------------------------------------------------------------
    [~,ndxv] = sort(sigma2draw,'ascend');
    
    if min(ndxv == (1:M)) == 0 % sigmas are not sorted
        indexSwitch=indexSwitch+1;
        betadraw=betadraw(:,ndxv);
        sigma2draw=sigma2draw(ndxv);
        Pdraw=Pdraw(ndxv,ndxv);
        p=p(ndxv);
        tm=tm(ndxv,ndxv);
        filprobdraw=filprobdraw(:,ndxv);
        sdraw_tmp = zeros(size(sdraw));
        for j=1:M
            if ~isempty(sdraw==ndxv(j))
                sdraw_tmp(sdraw==ndxv(j)) = j;
            end
        end
        sdraw = sdraw_tmp;
        
    end
    
    
    % -----------------------------------------------------------------------------------------
    %   Store draws, after burn-in period
    % -----------------------------------------------------------------------------------------
    if irep>burn
        
        %after discarding burnin, store all draws
        if mod(irep,thin_factor) == 0 % thinning of draws
            Beta_draws(:,(irep - burn)/thin_factor)   = betadraw(:); % note that reshape(betadraw,K,M) will bring it back to its original shape
            sigma2_draws((irep - burn)/thin_factor,:) = sigma2draw;
            P_draws(:,(irep - burn)/thin_factor)      = Pdraw(:); % note that reshape(Pdraw,M,M) will bring it back to its original shape
            Pr_draws(:,(irep - burn)/thin_factor,:)   = filprobdraw; 
            S_draws(:,(irep - burn)/thin_factor)      = sdraw;
            
        end
    end
end %end of gibb sampling loop

%Store the MCMC drawing into the structure result (trim the first
%obs)
%% Store results into output structure
result.beta     = Beta_draws';
result.sig2     = sigma2_draws;
result.P        = P_draws';
result.Pr       = Pr_draws;
result.S        = S_draws;

% Store data as well
result.y = EstY;
result.x = X;

fprintf('\r');