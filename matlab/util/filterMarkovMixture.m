function [lnl,filprob,simstate] = filterMarkovMixture(p,P,lnpdat,indSS)
% PURPOSE: filtering on mixture models 
% -----------------------------------------------------
% USAGE: [lnl,filprob,simstate] = filterMarkovMixture(p,P,lnpdat,indSS)
% where: 
% NOTE: no dimension checks imposed , 08/06/04
% -----------------------------------------------------
% RETURNS: 
% 
%
% -----------------------------------------------------
% written by:
% Gianni Amisano, Dept of Economics
% University of Brescia
% amisano@eco.unibs.it
% Modified by Davide Pettenuzzo
% Brandeis University
% dpettenu@brandeis.edu

T=size(lnpdat,1);
M=size(lnpdat,2);
simstate=0;
filprob=zeros(T,M);
lnl=zeros(T,1);

% Use steady-state probabilities to initialize the filtering algorithm
pit1t1=p;
for t=1:T
    % (a) Prediction step
    pitt1=P'*pit1t1;
    lnpmax=max(lnpdat(t,pitt1>0)); % this help avoiding numerical errors (for cases when y is very unlikely compared to the predictions from the model)
    if size(lnpmax,1)*size(lnpmax,2)~=1
        t
        lnpmax
        pitt1
        P
        pit1t1
        error('Something wrong in Hamilton prediction step');
    end
    % (b) Filtering step
    pitt=pitt1.*(exp(lnpdat(t,:)-lnpmax)');
    lkl=sum(pitt);
    if lkl==0
        pitt1
        lnpdat1(t,:)
        error('Something wrong in Hamilton filtering step');
    end
    pitt=pitt/lkl;
    lnl(t)=log(lkl)+lnpmax;   
    filprob(t,:)=pitt';
    pit1t1=pitt;
end

% Now implement multi-move sampling of the states. see Algorithm 11.5 on
% pages 342-343 of Fruhwirth-Schnatter (2006)
if indSS~=0
    simstate=zeros(T,1);
    simstate(T)=draw_multinom(filprob(T,:)',1,M,1); % this is the same as % simstate(T)=find(mnrnd(1,filprob(T,:)',1)==1);
    for t=(T-1):-1:1
        p1=(filprob(t,:)').*P(:,simstate(t+1));
        p1=p1/sum(p1);
        simstate(t)=draw_multinom(p1,1,M,1); % this is the same as simstate(t)=find(mnrnd(1,p1,1)==1);
    end
end