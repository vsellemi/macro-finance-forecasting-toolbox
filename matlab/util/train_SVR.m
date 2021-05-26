% Description: this function trains a support vector regression model

% Author: Victor Sellemi

% INPUT: 
%        - X      = (n x k) matrix of regressors
%        - y      = (n x 1) vector of dependent variables
%        - sigma  = scalar parameter for RBF kernel
%        - kernel = specifies kernel function (default is RBF)
%        - eps, C = scalar hyperparameters for SVR
% OUTPUT: 
%        - SVR    = support vector regression model in a structure 


function SVR = train_SVR(X,Y,sigma,kernel,eps,C)

    if nargin < 4
        kernel = 'gaussian';
        eps    = 0.01;
        C      = 1; 
    end

    Ntr = size(X,1); 
    Ktr = zeros(Ntr, Ntr); 
    
    % kernel calculation    
    switch kernel 
        case 'gaussian'
            XXh1 = sum(X.^2,2) * ones(1,size(X,1)); 
            XXh2 = sum(X.^2,2) * ones(1,size(X,1)); 
            D  = XXh1 + XXh2' - 2*(X*X');
            Ktr = exp(-D/(2*sigma^2))';
        case 'linear'
            Ktr = X * X';
        case 'poly2'
            Ktr = (X * X').^2;
        case 'poly3'
            Ktr = (X * X').^3;
        case 'sigmoid'
            Ktr = tanh(X * X');
    end
    

    % quadratic optimization setup (see p. 436 Hastie, Tibshirani, Friedman(2008))
    options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex',...
        'Display', 'off'); 
    
    % use sparse arrays for speed     
    H   = sparse(0.5*[Ktr + (1/C) * eye(Ntr), zeros(Ntr, 3*Ntr); zeros(3*Ntr, 4*Ntr)]); 
    f   = sparse([-Y; eps * ones(Ntr,1); zeros(2*Ntr,1)]); 
    lb  = sparse([-C * ones(Ntr,1); zeros(Ntr,1); zeros(2*Ntr,1)]); 
    ub  = sparse([C * ones(Ntr,1); 2*C*ones(Ntr,1); C*ones(2*Ntr,1)]);
    sol = quadprog(H,f,[],[],[],[],lb,ub,[],options); 
    
    % solution
    alpha = sol(1:Ntr); 
    
    % calculate bias
    bmat = zeros(Ntr,1); 
    H    = Ktr + (1/C) * eye(Ntr);
    for i = 1:Ntr
        bmat(i) = Y(i) - sum(alpha'.*H(i,:)) - eps - alpha(i) / C;
    end
    b = mean(bmat); 
    
    SVR = struct('alpha',alpha,'b',b,'X',X,'kernel',kernel,'sigma',sigma);     
    
end