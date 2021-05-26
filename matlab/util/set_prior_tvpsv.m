function prior = set_prior_tvpsv(Xtr, Ytr, LAM_mean, LAM_var, k_xsi, v_xsi, k_h, ...
    GAM_mean, GAM_var, psi_mat,v0_mat, k_Q, v_Q)

% Priors on regression coefficients depends on what the benchmark is in the different cases
%           this implies that the best forecast for Y this period is the h-period before Y (no change model)
prior.b0 = zeros(size(Xtr,2),1);
RHS = Xtr; Est_Y = Ytr;

s02                = ((Est_Y-RHS*prior.b0)'*(Est_Y-RHS*prior.b0))/(size(Est_Y,1)-size(RHS,2)); % residuls variance computed under the assumption true model for Y is RW
prior.V0           = (psi_mat^2) * (s02*(RHS'*RHS)\eye(size(RHS,2)));
prior.v0           = v0_mat*size(Est_Y,1);

prior.h_prmean     = log(sqrt(s02));
prior.h_prvar      = k_h;
prior.k_xsi        = k_xsi^2;
prior.v_xsi        = v_xsi*size(Est_Y,1);

prior.lam_mean     = LAM_mean;
prior.lam_var      = LAM_var;

prior.gam_mean     = GAM_mean*ones(size(RHS,2),1);
prior.gam_var      = GAM_var*eye(size(RHS,2));

% Note that for IW distribution I keep the _prmean/_prvar notation....
% Q is the covariance of theta(t)
% Q ~ IW(k_Q^2*size(subsample)*Var(B_OLS),size(subsample))

prior.Q_prmean = (k_Q^2)*round(v_Q*size(Est_Y,1))*prior.V0;
prior.Q_prvar  = round(v_Q*size(Est_Y,1));

end