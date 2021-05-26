function f = gamm_pdf_DP (y, mu, v)
% PURPOSE: returns the pdf at y of the gamma(mu,v) distribution (see
% Poirier (1995), p.98 and Koop et al (2007), pp.335-336
%---------------------------------------------------
% USAGE: pdf = gamm_pdf(y,mu,v)
% where: x = a vector  
%        mu = a scalar for gamma(mu,v)
%        v  = a scalar for gamma(mu,v)
%---------------------------------------------------
% RETURNS:
%        a vector of pdf at each element of y of the gamma(mu,v) distribution      
% --------------------------------------------------
% SEE ALSO: gamm_cdf, gamm_rnd, gamm_inv
%---------------------------------------------------

%       Anders Holtsberg, 18-11-93
%       Copyright (c) Anders Holtsberg
%       Modified by Davide Pettenuzzo, 04-14-2014

if nargin ~= 3
error('Wrong # of arguments to gamm_pdf_DP');
end;

if any(any(mu<=0))
   error('gamm_pdf_DP: parameter mu is wrong')
end

c_G = ((2*mu/v)^(v/2)) * gamma(v/2);
f   = y .^ ((v-2)/2) .* exp(-y*v/(2*mu)) ./ c_G;
I0 = find(y<0);
f(I0) = zeros(size(I0));
