function [c,u,l]=bsds(bench,models,B,w,type,boot)
% Calculate Whites and Hansens p-vals for out-performance using unmodified data or studentized
% residuals,  the latter often providing better power, particularly when the losses functions are
% heteroskedastic
%
% USAGE:
%   [C] = bsds(BENCH,MODELS,B,W)
%   [C,U,L] = bsds(BENCH,MODELS,B,W,TYPE,BOOT)
%
% INPUTS:
%   BENCH  - Losses from the benchmark model
%   MODELS - Losses from each of the models used for comparison
%   B      - Number of Bootstrap replications
%   W      - Desired block length
%   TYPE   - String, either 'STANDARD' or 'STUDENTIZED'.  'STUDENTIZED' is the default, and
%              generally leads to better power.
%   BOOT   - [OPTIONAL] 'STATIONARY' or 'BLOCK'.  Stationary is used as the default.
%
% OUTPUTS:
%   C      - Consistent P-val(Hansen)
%   U      - Upper P-val(White) (Original RC P-vals)
%   L      - Lower P-val(Hansen)
%
% COMMENTS:
%   This version of the BSDS operates on quantities that should be 'bads', such as losses.  The null
%   hypothesis is that the average performance of  the benchmark is as small as the minimum average
%   performance across the models.  The alternative is that the minimum average loss across the
%   models is smaller than the the average performance of the benchmark.
%
%   If the quantities of interest are 'goods', such as returns, call bsds with
%   -1*BENCH and -1*MODELS
%
% EXAMPLES:
%   Standard Reality Check with 1000 bootstrap replications and a window size of 12
%       bench = randn(1000,1).^2;
%       models = randn(1000,100).^2;
%       [c,realityCheckPval] = bsds(bench, models, 1000, 12)
%   Standard Reality Check with 1000 bootstrap replications, a window size of 12 and a circular
%   block bootstrap
%       [c,realityCheckPval] = bsds(bench, models, 1000, 12, 'BLOCK')
%   Hansen's P-values
%       SPAPval = bsds(bench, models, 1000, 12)
%   Both Pvals on "goods"
%       bench = .01 + randn(1000,1);
%       models = randn(1000,100);
%       [SPAPval,realityCheckPval] = bsds(-bench, -models, 1000, 12)
%
% See also MCS

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 3    Date: 4/1/2007

%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Checking
%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<4 || nargin>6
    error('4 to inputs required')
end
if nargin == 4
    boot = 'STATIONARY';
    type = 'STUDENTIZED';
elseif nargin == 5
    boot = 'STATIONARY';
end
if isempty(type)
    type = 'STUDENTIZED';
end
if strcmpi(type,'STUDENTIZED')
    isStudentized = true;
else
    isStudentized = false;
end
% Get the length of the data
[tb,kb]=size(bench);
if kb>1
    error('BENCH must be a column vector')
end
if tb<2
    error('BENCH must have at least 2 observations.')
end
[t,k]=size(models);
if t~=tb
    error('BENCH and MODELS must have the same number of observations.')
end
if ~isscalar(B) || B<1 || floor(B)~=B
    error('B must be a positive scalar integer')
end
if ~isscalar(w) || w<1 || floor(w)~=w
    error('W must be a positive scalar integer')
end
boot = upper(boot);
if ~ismember(boot,{'STATIONARY','BLOCK'})
    error('BOOT must be either ''STATIONARY'' or ''BLOCK''.')
end
%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Checking
%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(boot,'BLOCK')
    bsdata=block_bootstrap((1:t)',B,w);
else
    bsdata=stationary_bootstrap((1:t)',B,w);
end

%OK now we have the bootstraps, what to do with them?
diffs=models-repmat(bench,1,k);

% First compute the boostarap sample averages, db*
% Second compute the variance estimate, omegak
% First the weghts
q=1/w;
i=1:t-1;
kappa=((t-i)./t).*(1-q).^i+i./t.*(1-q).^(t-i);
% Next compute the variances
vars=zeros(k,1)';
for i=1:k
    workdata = diffs(:,i)-mean(diffs(:,i));
    vars(i)= workdata'*workdata/t;
    for j=1:t-1
        vars(i) = vars(i) + 2*kappa(j)*(workdata(1:t-j)'*workdata(j+1:t))/t;
    end
end

% Aold is the original method to compute the truncation point
Aold=1/4*t^(0.25)*sqrt(vars/t);
mean(Aold);
% A new used the log(log(t)) rule
Anew = sqrt((vars/t)*2*log(log(t)));

% Only recenter if the average is reasonably small or the model is better
% (in which case mean(diffs) is negative).  If it is unreasonably large set
% the mean adjustment to 0
gc=mean(diffs).*(mean(diffs)<Anew);


% The lower assumes that every loss function that is worse than BM is
% unimportant for the asymptotic distribution, hence if its mean is
% less than 0, g=0.  This is different from the consistent where the
% threshold was it had to be greater than -A(i)
gl=min(0,mean(diffs));

%Then the upper, which assumes all models used are reasonably close to
%the benchmark that they coudl be better
gu=mean(diffs);

% Perf will hold the boostrapped statistics for B iterations
perfc=zeros(B,k);
perfl=zeros(B,k);
perfu=zeros(B,k);
if isStudentized
    stdDev = sqrt(vars);
else
    stdDev = ones(1,k);
end

for i=1:k
    workdata=diffs(:,i);
    % the i'th column of perf holds the B bootstrapped statistics
    mworkdata=mean(workdata(bsdata));
    perfc(:,i)=(mworkdata-gc(i))'/stdDev(i);
    perfl(:,i)=(mworkdata-gl(i))'/stdDev(i);
    perfu(:,i)=(mworkdata-gu(i))'/stdDev(i);
end
% Compute the test statistic
stat = min(mean(diffs)./stdDev);
% Compute the min in each row
perfc=min(perfc,[],2);
perfc=min(perfc,0);
perfl=min(perfl,[],2);
perfl=min(perfl,0);
perfu=min(perfu,[],2);
perfu=min(perfu,0);
% Count the number of time the min is below the statistic
c=mean(perfc<stat);
l=mean(perfl<stat);
u=mean(perfu<stat);