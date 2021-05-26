function draw = draw_multinom(p,T,m,indE)
% PURPOSE: filtering on mixture models 
% -----------------------------------------------------
% USAGE: draw = draw_multinom(p,T,m,indE)
% where: 
% p is the vector of probabilities
% T is the number of draws we want
% m is the number of different states
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
if indE==0
   pp=cumsum(p,2);    
else
    pp=repmat((cumsum(p))',T,1);
end
[d1,d2]=max(repmat(rand(T,1),1,m)<pp,[],2);
draw=d2;