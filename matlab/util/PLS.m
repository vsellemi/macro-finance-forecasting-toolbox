% Description: this function implements partial least squares regression
% Author: Victor Sellemi

% INPUT: 
%        - X   = (n x k) matrix of regressors
%        - Y   = (n x 1) vector of dependent variables
%        - tol = error tolerance for convergence (default 1e-10)
% OUTPUT: 
%        - S   = structure of results with elements T,P,U,Q,B,W such that:
%                           -- X = T*P' + E
%                           -- Y = U*Q' + F = T*B*Q' + F1
% NOTE: 
%        - To predict for a new X1 use the formula:
%               Y1hat = (X1*P)*B*Q' = X1*(P*B*Q')
%        - When Y is not provided, P' represents the principal components
%        of X such that X = T*P'+E


function S = PLS(X,Y,tol2)

if nargin < 2
    Y = X;
end

tol = 1e-10;
if nargin < 3
    tol2 = 1e-10;
end

[Tx,kx]  =  size(X);
[Ty,ky]  =  size(Y);

assert(Tx == Ty,'Sizes of X and Y mismatch.');

% preallocate
k = 0;
n = max(kx,ky); 
T = zeros(Tx,n); 
P = zeros(kx,n); 
U = zeros(Ty,n);
Q = zeros(ky,n);
B = zeros(n,n);
W = P;

while norm(Y) > tol2 && k < n
    
    % choose t and u as largest sum of squares in X and Y respectively
    [~,tidx] =  max(sum(X.*X));
    [~,uidx] =  max(sum(Y.*Y));
    
    t1 = X(:,tidx);
    u  = Y(:,uidx);
    t  = zeros(Tx,1);
    
    % iterate until convergence
    while norm(t1 - t) > tol
        w  = X' * u;
        w  = w / norm(w);
        t  = t1;
        t1 = X * w;
        q  = Y'*t1;
        q  = q / norm(q);
        u  = Y * q;
    end
    
    % update p 
    t     = t1;
    p     = X' * t / (t'*t);
    pnorm = norm(p);
    p     = p / pnorm;
    t     = t * pnorm;
    w     = w * pnorm;
    
    % residuals
    b = u'*t / (t'*t);
    X = X - t * p';
    Y = Y - b * t * q';
    
    k=k+1;
    T(:,k)=t;
    P(:,k)=p;
    U(:,k)=u;
    Q(:,k)=q;
    W(:,k)=w;
    B(k,k)=b;

end

T(:,k+1:end)=[];
P(:,k+1:end)=[];
U(:,k+1:end)=[];
Q(:,k+1:end)=[];
W(:,k+1:end)=[];
B=B(1:k,1:k);

S = struct('T',T,'P',P,'U',U,'Q',Q,'W',W,'B',B); 

end