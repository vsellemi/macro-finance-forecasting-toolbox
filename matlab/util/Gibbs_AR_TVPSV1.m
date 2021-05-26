function result=Gibbs_AR_TVPSV(EstY,EstX,prior_struct,I,burn,thin_factor)


% Gibbs sampling with independent Normal-Gamma prior 

% Linear normal model with y_t=mu + rho*y(lagged) + alpha*Exogenous_t 
%                              + eps_t where eps_t~iid N(0,sig2)
% Priors for parameters are:
%                           [mu;rho;alpha]~ N(b0,V0)
%                           (1/sig2)            ~ G(1/s02,v0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% EstY          : dependent variable (T x 1)
% EstX          : exogenous X variables lags (T x K)
% EstLagY       :lags of dependent variable(T x k)
% prior_struct  : prior hyperparameters
% I             : total number of iterations for Gibbs sampler
% burn          : number of burn in draws  for Gibbs sampler
% thin_factor   : frequency of draws being retained  for Gibbs sampler
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Pull out priors from prior structure
field_list = fieldnames(prior_struct);

for i=1:length(field_list)
    eval([char(field_list(i)),' = getfield(prior_struct,''',char(field_list(i)),''');']);
end

% Parameters of the 7 component mixture approximation to a log(chi^2)
% density (from Kim et al. (1998):
q_s  = [0.00730; 0.10556; 0.00002; 0.04395; 0.34001; 0.24566; 0.25750];     % probabilities
m_s  = [-10.12999; -3.97281; -8.56686; 2.77786; 0.61942; 1.79518; -1.08819];% means
u2_s = [5.79596; 2.61369; 5.17950; 0.16735; 0.64009; 0.34023; 1.26261];    % variances

%% Prepare X matrix for regression
[T,~] = size(EstY);

% Combine lagged Y and lagged exogenous regressors
X     = EstX;
[~,k] = size(X);

%% Ordinary least squares quantities (will be used to initialize gibbs
%  sampler)
bols = (X'*X)\(X'*EstY);
s2 = (EstY-X*bols)'*(EstY-X*bols)/(T-k);

% INITIALIZE MATRICES:
% First the regression coefficients
Bdraw = (X'*X)\(X'*EstY);
Btdraw = zeros(k,T);
htdraw = zeros(T,1);   % Initialize Sigmatdraw, a draw of the volatility states
% initialize the parameters of the AR process for ht
lam_draw = [0;1];

% initialize the parameters of the AR process for TVP
gam_draw = eye(size(X,2));

% Next the prior hyperparameter
consW = 0.0001;
sigma2_xsidraw = consW;    

% finally the prior hyperparameters
consQ = 0.0001;
Qdraw = consQ*eye(k);   % Initialize Qdraw, a draw from the covariance matrix Q


% Need also to draw the states (added to the chain to permit drawing the
% realized volatilities using standard linear gaussian Kalman fiter
statedraw = 5*ones(T,1);       % initialize the draw of the indicator variable 

% These are additional matrices needed to draw the volatility states
% (of 7-component mixture of Normals approximation)
Zs = 2*ones(T,1);
prw = zeros(numel(q_s),1);


% Calculate a few quantities outside the loop for later use
capv0inv = V0\eye(size(V0,1));
v1=v_xsi+(T-1);
v0s02=v_xsi*k_xsi;

capv0inv_lam = lam_var\eye(size(lam_var));
capv0inv_gam = gam_var\eye(size(gam_var));


%% Start Gibbs sampler

%store all draws in the following matrices
Beta_draws       = NaN(size(X,2),(I-burn)/thin_factor);
Beta_t_draws     = NaN(T,(I-burn)/thin_factor,size(X,2));
h_t_draws        = NaN(T,(I - burn)/thin_factor);
lam_draws        = NaN(2,(I - burn)/thin_factor);
gam_draws        = NaN(size(X,2),(I - burn)/thin_factor);
sigma2_xsi_draws = NaN((I - burn)/thin_factor,1);
Q_draws          = NaN(size(X,2)^2,(I-burn)/thin_factor,1);


for irep = 1:I
    
    % Print iterations
    if mod(irep,100) == 0
        if irep == 100
            fprintf('\r Iterations: %d \t',irep)
        else
            fprintf('%d \t',irep);
        end
    end
    
    
    %------------------------------------------------------------------------------------------
    %   STEP I: Draw all h(t)
    %------------------------------------------------------------------------------------------
    % Create Y_start2, transforming to the log-scale but also adding the 'offset constant' to prevent
    % the case where Y_star is zero (in this case, its log would be -Infinity)
    y_star  = NaN(T,1);
    for i = 1:T
        y_star(i,:) = EstY(i,:) - X(i,:)*Bdraw - X(i,:)*Btdraw(:,i);
    end
    y_star2 = log(y_star.^2 + 0.001);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATED BY MOVING THIS BLOCK UP RELATIVE TO THE ORIGINAL PRIMICERI
    % 2005 ALGORITHM - SEE DEL NEGRO PRIMICERI CORRECTION 2014
    % Next draw statedraw (chi square approximation mixture component) conditional on Sigtdraw
    % This is used to update at the next step the log-volatilities Sigtdraw
    for tau = 1:T
        for j = 1:numel(m_s)
            temp1= (1/sqrt(2*pi*u2_s(j)))*exp(-.5*(((y_star2(tau) - 2*htdraw(tau) - m_s(j) + 1.2704)^2)/u2_s(j)));
            prw(j,1) = q_s(j,1)*temp1;
        end
        prw = prw./sum(prw);
        cprw = cumsum(prw);
        trand = rand(1,1);
        if trand < cprw(1,1); imix=1;
        elseif trand < cprw(2,1), imix=2;
        elseif trand < cprw(3,1), imix=3;
        elseif trand < cprw(4,1), imix=4;
        elseif trand < cprw(5,1), imix=5;
        elseif trand < cprw(6,1), imix=6;
        else imix=7;
        end
        statedraw(tau)=imix;  % this is a draw of the mixture component index
    end
    
    % In order to draw the log-volatilies, substract the mean and variance
    % of the 7-component mixture of Normal approximation to the measurement
    % error covariance
    vart = zeros(T,1);
    yss1 = zeros(T,1);
    for j = 1:T
        imix      = statedraw(j);
        vart(j)   = u2_s(imix);
        yss1(j) = y_star2(j) - m_s(imix) + 1.2704;
    end
    
    % htdraw is a draw of the diagonal elements 2*log(h(t))
    [htdraw,hTT,VTT] = carter_kohn(yss1',Zs,vart,lam_draw(1),lam_draw(2),sigma2_xsidraw,1,1,T,h_prmean,h_prvar);
    htdraw = htdraw';
    
    % Draws in htdraw are in logarithmic scale (log-volatilies). Create
    % original standard deviations of return equation
    Sigmat = exp(htdraw);
    
    % -----------------------------------------------------------------------------------------
    %   STEP II: Sample W from p(W|y,B_t,Sigma) which is i-Gamma
    % -----------------------------------------------------------------------------------------
    % Get first differences of htdraw to compute the SSE
    sse_2          = (htdraw(2:T) - repmat(lam_draw(1),T-1,1) - htdraw(1:T-1)*lam_draw(2))'*...
                     (htdraw(2:T) - repmat(lam_draw(1),T-1,1) - htdraw(1:T-1)*lam_draw(2));
    s12            = (sse_2 + v0s02)/v1;
    sigma2_xsidraw = 1/gamm_rnd(1,1,.5*v1,.5*v1*s12);
    
    % -----------------------------------------------------------------------------------------
    %   STEP III: Sample Lam_0 and Lam_1
    % -----------------------------------------------------------------------------------------
    tmp_1 = 0;
    tmp_2 = 0;
    for j=2:T
        tmp_1 = tmp_1 + (1/sigma2_xsidraw)*([1,htdraw(j-1)]'*[1,htdraw(j-1)]);
        tmp_2 = tmp_2 + (1/sigma2_xsidraw)*([1,htdraw(j-1)]'*htdraw(j));
    end
    
    lam_post_var  = (capv0inv_lam + tmp_1)\eye(size(lam_var));
    lam_post_mean = lam_post_var*(capv0inv_lam*lam_mean + tmp_2);
    P = chol(lam_post_var);
    lam_draw = lam_post_mean + P'*randn(size(lam_var,1),1);
    
    % Impose stationarity by adding an accept-reject checking on
    % lam_draw(2)
    while lam_draw(2) <= 0 || lam_draw(2) >= 1
        lam_draw = lam_post_mean + P'*randn(size(lam_var,1),1);
    end
    
    % -----------------------------------------------------------------------------------------
    %   STEP IV: Sample beta_t from p(theta_t|y,theta,Sigma) 
    % -----------------------------------------------------------------------------------------
    
    % Create Y_t_star, Z_t, and W_t matrices
    Y_t_star = NaN(T,1);
    for i = 1:T
        Y_t_star(i,:) = EstY(i,:) - X(i,:)*Bdraw;
    end
    
    [Btdrawc,BTT,PTT] = carter_kohn(Y_t_star',X,Sigmat.^2,zeros(size(X,2),1),gam_draw,Qdraw,k,1,T);
    % Accept draw (no issues of stationarity)
    Btdraw = Btdrawc;
        
    % -----------------------------------------------------------------------------------------
    %   STEP V: Sample theta from p(theta|y,theta_t,Sigma) 
    % -----------------------------------------------------------------------------------------
    % Create Y_t_star2 matrix
    Y_t_star2 = NaN(T,1);
    tmp_1 = 0;
    tmp_2 = 0;
    
    for i = 1:T
        Y_t_star2(i,:) = EstY(i,:) - X(i,:)*Btdraw(:,i);
        tmp_1 = tmp_1 + X(i,:)'*(1/(Sigmat(i)^2))*X(i,:);
        tmp_2 = tmp_2 + X(i,:)'*(1/(Sigmat(i)^2))*Y_t_star2(i,:);
    end
    
    B_post_var  = (capv0inv + tmp_1)\eye(k);
    B_post_mean = B_post_var*(capv0inv*b0 + tmp_2);
    
    P = chol(B_post_var);
    Bdraw = B_post_mean + P'*randn(k,1);
    
    % -----------------------------------------------------------------------------------------
    %   STEP VI: Sample Q from p(Q|y,theta_t,Sigma) which is i-Wishart
    % -----------------------------------------------------------------------------------------
    Btemp = Btdraw(:,2:T)' - Btdraw(:,1:T-1)'*gam_draw;
    sse_2Q = zeros(k,k);
    for i = 1:T-1
        sse_2Q = sse_2Q + Btemp(i,:)'*Btemp(i,:);
    end
    Qinv = inv(sse_2Q + Q_prmean);
    Qinvdraw = wish_rnd(Qinv,T+Q_prvar);
    Qdraw = Qinvdraw\eye(k);
    
    % -----------------------------------------------------------------------------------------
    %   STEP VII: Sample Gam
    % -----------------------------------------------------------------------------------------
    for ii=1:size(X,2)
        tmp_1 = 0;
        tmp_2 = 0;
        for j=2:T
            tmp_1 = tmp_1 + (1/Qdraw(ii,ii))*(Btdraw(ii,j-1)'*Btdraw(ii,j-1));
            tmp_2 = tmp_2 + (1/Qdraw(ii,ii))*(Btdraw(ii,j-1)'*Btdraw(ii,j));
        end
        
        gam_post_var  = 1/(capv0inv_gam(ii,ii) + tmp_1);
        gam_post_mean = gam_post_var*(capv0inv_gam(ii,ii)*gam_mean(ii) + tmp_2);
        P = chol(gam_post_var);
        gam_draw(ii,ii) = gam_post_mean + P'*randn;
        
        % Impose stationarity by adding an accept-reject checking on
        % gam_draw
        while gam_draw(ii,ii) <= 0 || gam_draw(ii,ii) >= 1
            gam_draw(ii,ii) = gam_post_mean + P'*randn;
        end
    end
    
    % -----------------------------------------------------------------------------------------
    %   Store draws, after burn-in period
    % -----------------------------------------------------------------------------------------
    if irep>burn
        
        %after discarding burnin, store all draws
        if mod(irep,thin_factor) == 0 % thinning of draws
            Beta_draws(:,(irep - burn)/thin_factor)     = Bdraw;
            Beta_t_draws(:,(irep - burn)/thin_factor,:) = Btdraw';
            h_t_draws(:,(irep - burn)/thin_factor,:)    = htdraw;
            lam_draws(:,(irep - burn)/thin_factor)      = lam_draw;
            sigma2_xsi_draws((irep - burn)/thin_factor) = sigma2_xsidraw;
            gam_draws(:,(irep - burn)/thin_factor)      = diag(gam_draw); % note that reshape(gam_draw,k,k) will bring it back to its original shape
            Q_draws(:,(irep - burn)/thin_factor)        = Qdraw(:); % note that reshape(Qdraw,k,k) will bring it back to its original shape
        end
    end
end

%% Store results into output structure
result.beta     = Beta_draws';
result.beta_t   = Beta_t_draws;
result.h_t      = h_t_draws;
result.lam      = lam_draws';
result.sig2_xsi = sigma2_xsi_draws;
result.gam      = gam_draws';
result.Q        = Q_draws';

% Store data as well
result.y = EstY;
result.x = X;

fprintf('\r');
