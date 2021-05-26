function []=FanChart(X,Y)

% This function generates a fan chart. A fan chart aims to visualize the
% uncertainty that surrounds a sequence of point forecasts.
%
% Output:
% -- plot


T=size(Y,1);

% Set relevant quantiles (in percent)
% quantiles=[5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95];
  quantiles=[5 10       25             50             75       90 95];
num_quant=size(quantiles,2);

% Compute quantiles
mat_quant=NaN(T,num_quant);
for h=1:T % loop over time
    for q=1:num_quant % loop over quantiles
        mat_quant(h,q)=quantile(Y(h,:),quantiles(1,q)/100);
    end
end

% Prepare increment matrix for its use with the area function
matm=mat_quant;
for i=2:size(matm,2)
    matm(:,i)=matm(:,i)-mat_quant(:,i-1);
end

% Find NaNs and fix them before the plot
nan_indx = sum(isnan(matm),2)>1;
matm(nan_indx,:) = 0;

% Generate plot
h=area(X,matm);
r=.2;
b=0;

% set(h,'LineStyle','none')
% set(h(1),'FaceColor',[1 1 1]) % white
% set(h(2),'FaceColor',[r .99 b])
% set(h(3),'FaceColor',[r .975 b])
% set(h(4),'FaceColor',[r .95 b])
% set(h(5),'FaceColor',[r .9 b])
% set(h(6),'FaceColor',[r .85 b])
% set(h(7),'FaceColor',[r .8 b])
% set(h(8),'FaceColor',[r .75 b])
% set(h(9),'FaceColor',[r .7 b])
% set(h(10),'FaceColor',[r .65 b]) %
% set(h(11),'FaceColor',[r .65 b])%
% set(h(12),'FaceColor',[r .7 b])
% set(h(13),'FaceColor',[r .75 b])
% set(h(14),'FaceColor',[r .8 b])
% set(h(15),'FaceColor',[r .85 b])
% set(h(16),'FaceColor',[r .9 b])
% set(h(17),'FaceColor',[r .95 b])
% set(h(18),'FaceColor',[r .975 b])
% set(h(19),'FaceColor',[r .99 b])

set(h,'LineStyle','none')
set(h(1),'FaceColor',[1 1 1]) % white
set(h(2),'FaceColor',[r .99 b])
% set(h(3),'FaceColor',[r .975 b])
% set(h(4),'FaceColor',[r .95 b])
set(h(3),'FaceColor',[r .85 b])
% set(h(6),'FaceColor',[r .85 b])
% set(h(7),'FaceColor',[r .8 b])
% set(h(8),'FaceColor',[r .75 b])
% set(h(9),'FaceColor',[r .7 b])
set(h(4),'FaceColor',[r .65 b]) %
% set(h(11),'FaceColor',[r .65 b])%
% set(h(12),'FaceColor',[r .7 b])
% set(h(13),'FaceColor',[r .75 b])
% set(h(14),'FaceColor',[r .8 b])
set(h(5),'FaceColor',[r .65 b])
% set(h(16),'FaceColor',[r .9 b])
% set(h(17),'FaceColor',[r .95 b])
set(h(6),'FaceColor',[r .85 b])
set(h(7),'FaceColor',[r .99 b])
