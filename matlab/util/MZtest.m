function [MZstat MZpval]=MZtest(y,yhat)
   
  for ii=1:size(yhat,2)
    intercept=ones(length(y),1); X=[intercept yhat(:,ii)];
    [x,resnormU] = lsqlin(X,y); SSE=sum(resnormU);
    A=eye(2); b=[0;1]; N_restriction=length(b); [x,resnormR] = lsqlin(X,y,A,b); SSER=sum(resnormR);
    N=length(y); k=size(X,2); df=N-k; %degrees of freedom  
    MZstat(ii,1)=((SSER-SSE)/N_restriction)/(SSE/df);
    MZpval(ii,1)=1-fcdf(MZstat(ii,1),N_restriction,N-k);
  end
   
end


% %F-test
%   %Simulated Data
% T=1000;
% ResidualVar=0.1;
% beta=1;
% alfa=0;
% yhat=normrnd(0,1,T,1); y=alfa*ones(T,1)+yhat*beta+normrnd(0,ResidualVar,T,1);
%   %Real Data
% 
% scatter(yhat,y)
% intercept=ones(length(y),1); X=[intercept yhat];
% %Unrestricted
% [x,resnormU] = lsqlin(X,y); SSE=sum(resnormU);
% %Restricted
% A=eye(2); b=[0;1]; N_restriction=length(b); [x,resnormR,residual] = lsqlin(X,y,A,b); SSER=sum(resnormR);
% %Compute MZtest
% N=length(y); k=size(X,2); df=N-k; %degrees of freedom
% Ftest=((SSER-SSE)/N_restriction)/(SSE/df);
% pval=1-fcdf(Ftest,N_restriction,N-k);
