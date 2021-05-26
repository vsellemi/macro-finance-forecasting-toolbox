function prior = set_prior_ms(Xtr,Ytr,psi_mat,v0_mat,ekk,ekkp,K)

Est_Y = Ytr; 


% Combine constant, lagged Y and lagged exogenous regressors
RHS     = Xtr;

% Priors on regression coefficients depends on what the benchmark is in
% the different cases
prior.b0 = zeros(size(RHS,2),1); % this implies that the best forecast for inflation this period is the h-period before inflation (no change model)

prior.s02   = ((Est_Y-RHS*prior.b0)'*(Est_Y-RHS*prior.b0))/(size(Est_Y,1)-size(RHS,2)); % residuls variance computed under the assumption true model for Y is RW
prior.V0    = (psi_mat^2) * (prior.s02*(RHS'*RHS)\eye(size(RHS,2)));
prior.v0    = (v0_mat*size(Est_Y,1));

% Transition matrix probabilities
prior.Pmat  = eye(K)*ekk + (1-eye(K))*(ekkp/(K-1));

end