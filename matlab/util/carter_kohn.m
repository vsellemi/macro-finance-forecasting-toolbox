function [bdraw,btt_out,Vtt_out] = carter_kohn(y,Z,Sigmat,mu_til,F,Qt,m,p,t,varargin)
% Carter and Kohn (1994), On Gibbs sampling for state space models.

% Check number of arguments
if nargin == 9
    % This is the case where beta_0=0
    bp = zeros(m,1);
    Vp = zeros(m,m);
elseif nargin == 11
    % This is the case where we have priors for beta_0
    B0 = varargin{1};
    V0 = varargin{2};
    
    bp = B0;
    Vp = V0;
else
    error('Wrong number of inputs to carter_kohn_hom -- check your codes');
end

% Kalman filter notation:
% y(t) = H(t)*b(t)         + eps(t)
% b(t) = mu_til + F*b(t-1) + u(t)
%  _      _      _             _
% | eps(t) | ~N | 0  , R(t) 0   |
% |_ u(t) _|    |_0    0    Q  _|
% 

% Kalman Filter forward recursion starts here
bt = zeros(t,m);
Vt = zeros(m^2,t);

for i=1:t
    R = Sigmat((i-1)*p+1:i*p,:);
    H = Z((i-1)*p+1:i*p,:);
    
    % Prediction step
    bttm1 = mu_til +F*bp;
    Vttm1 = F*Vp*F'+Qt;
    
    % Updating step
    Kt = Vttm1*H'*((H*Vttm1*H' + R)\eye(size(R,1))); % Kalman gain
    
    btt = bttm1 + Kt*(y(:,i) - H*bp);
    Vtt = Vttm1 - Kt*H*Vttm1;
    
    if i < t
        bp = btt;
        Vp = Vtt;
    end
    bt(i,:) = btt';
    Vt(:,i) = reshape(Vtt,m^2,1);
end

% draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
bdraw = zeros(t,m);
bdraw(t,:) = mvnrnd(btt,Vtt,1);

btt_out = btt;
Vtt_out = Vtt;

% Kalman Filter backward recursion starts here
for i=1:t-1
    bf = bdraw(t-i+1,:)';
    btt = bt(t-i,:)';
    Vtt = reshape(Vt(:,t-i),m,m);
    inv_f = (F*Vtt*F' + Qt)\eye(size(Qt,1)); % this is V(t+1|t)^(-1)
    bmean = btt + Vtt*F'*inv_f*(bf - mu_til - F*btt);
    bvar = Vtt - Vtt*F'*inv_f*F*Vtt;
    bdraw(t-i,:) = bmean' + randn(1,m)*chol(bvar); % mvnrnd(bmean,bvar,1);
end
bdraw = bdraw';