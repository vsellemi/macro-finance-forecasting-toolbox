function [Pdraw,pv,tm] = MSLinRegDrawP(T,M,Sdraw,Prior_P)
% PURPOSE: draws transition probs in a MS lin reg model
% -----------------------------------------------------
% USAGE: [Pdraw,pv,tm] = MSLinRegDrawP(T,M,Sdraw,Prior_P)
% where: 
% NOTE: no dimension checks imposed , 04/10/04
% -----------------------------------------------------
% RETURNS: 
% 
%
% -----------------------------------------------------
% written by:
% Gianni Amisano, Dept of Economics
% University of Brescia
% amisano@eco.unibs.it
% Modified by Davide Pettenuzzo (commented out original lines being replaced)
% Brandeis University
% dpettenu@brandeis.edu

Pdraw=zeros(M,M);
sm=[Sdraw(1:T-1) Sdraw(2:T)];
tm=zeros(M,M);
for im=1:M
    for jm=1:M
        tm(im,jm)=size(sm(sm(:,1)==im & sm(:,2)==jm,:),1);              
    end
    Pdraw(im,:) = dirichlet_rnd(1,Prior_P(im,:)+tm(im,:));
    % Pdraw(im,:)=draw_dirichlet(1,M,Prior_P(im,:)+tm(im,:),0);  
end

% Compute the invariant distribution of S_t
pv=[(eye(M)-Pdraw'); ones(1,M)];
test=det(pv'*pv);
if test~=0
    pv=inv(pv'*pv)*pv';
    pv=pv(:,M+1);
else
    pv=ones(M,1)/M;
end
