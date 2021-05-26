function x = truncated_normal_ab_sample ( mu, s, a, b)

%*****************************************************************************80
%
%% TRUNCATED_NORMAL_AB_SAMPLE samples the truncated Normal PDF.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    14 August 2013
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real MU, S, the mean and standard deviation of the
%    parent Normal distribution.
%
%    Input, real A, B, the lower and upper truncation limits.
%
%    Output, real X, a sample of the PDF.
%
  alpha = ( a - mu ) / s;
  beta = ( b - mu ) / s;

  alpha_cdf = normcdf( alpha );
  beta_cdf  = normcdf( beta );

  xi_cdf = alpha_cdf + rand * ( beta_cdf - alpha_cdf );
  xi = norminv ( xi_cdf );

  x = mu + s * xi;

  return
end
