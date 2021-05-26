% Description: this function estimates the alpha vector in kernel ridge
% regression
%                   alpha = (K - lambda I_T)^{-1} * y

% Author: Victor Sellemi

% INPUT: 
%        - X      = (n x k) matrix of regressors
%        - y      = (n x 1) vector of dependent variables
%        - lambda = scalar ridge parameter
%        - sigma  = parameter in RBF kernel
%        - kernel = specifies kernel function (default is gaussian)
% OUTPUT: 
%        - alpha = (n x 1) vector 


function alpha = train_KRR(X,Y,lambda,sigma,kernel)

    if nargin < 5
        kernel = 'gaussian';
    end

    Ntr = size(X,1); 
    Ktr = zeros(Ntr, Ntr); 
    
    switch kernel 
        case 'gaussian'
            XXh1 = sum(X.^2,2) * ones(1,size(X,1)); 
            XXh2 = sum(X.^2,2) * ones(1,size(X,1)); 
            D  = XXh1 + XXh2' - 2*(X*X');
            Ktr = exp(-D/(2*sigma^2));
        case 'linear'
            Ktr = X * X';
        case 'poly2'
            Ktr = (X * X').^2;
        case 'poly3'
            Ktr = (X * X').^3;
        case 'sigmoid'
            Ktr = tanh(X * X');
    end
     
    alpha = pinv(Ktr + lambda*eye(Ntr)) * Y;

end