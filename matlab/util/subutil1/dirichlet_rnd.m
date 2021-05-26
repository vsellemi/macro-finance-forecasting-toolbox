function draw = dirichlet_rnd(N,a)
% PURPOSE: obtain n draws from K-variate dirichlet pdf
% -----------------------------------------------------
% USAGE: draw = draw_dirichlet(N,a)
% where: N is the number of draws needed
%        a is the parameters of the distribution 

% -----------------------------------------------------
% RETURNS: draw = (N x K) matrix with draws on different rows
% REFERENCE: Fruhwirth-Schnatter (2006), pages 432-433
% -----------------------------------------------------

K = length(a);
Y = NaN(N,K);
for i=1:K
    Y(:,i) = gamm_rnd(N,1,a(i),1);
end

draw = Y./repmat(sum(Y,2),1,K);
