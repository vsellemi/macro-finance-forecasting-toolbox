% Description: this function implements predictions from a trained kernel
% ridge regression model

% Author: Victor Sellemi

% INPUT: 
%        - X = (n x k) matrix of regressors
%        - y = (n x 1) vector of dependent variables
%        - lambda = scalar ridge parameter
%        - kernel = specifies kernel function: default is gaussian
%        - sigma
% OUTPUT: 
%        - alpha = (n x 1) vector 


function yhat = forecast_KRR(alpha,Xtr,Xte,sigma,kernel)

    if nargin < 5
        kernel = 'gaussian';
    end

    Ntr = size(Xtr,1);
    Nte = size(Xte,1); 
    K = zeros(Nte, Ntr); 
    
   switch kernel 
        case 'gaussian'
            XXh1 = sum(Xtr.^2,2) * ones(1,size(Xte,1));
            XXh2 = sum(Xte.^2,2) * ones(1,size(Xtr,1));
            D  = XXh1 + XXh2' - 2*Xtr*Xte';
            K = exp(-D/(2*sigma^2))';
        case 'linear'
            K = Xte * Xtr';
        case 'poly2'
            K = (Xtr * Xtr').^2;
        case 'poly3'
            K = (Xtr * Xtr').^3;
        case 'sigmoid'
            K = tanh(Xtr * Xtr');
   end
     
    yhat = K * alpha;

end