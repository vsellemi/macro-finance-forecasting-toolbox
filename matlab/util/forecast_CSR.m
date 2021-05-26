function yhat = forecast_CSR(X,S)

all_ind = S.idx;
betas   = S.betas;
yhats   = NaN(size(X,1),size(all_ind,1)); 

for c = 1:size(all_ind,1)
    tmpidx  = all_ind(c,:); 
    tmpBeta = betas(c,:)';
    yhats(:,c) = X(:,tmpidx) * tmpBeta; 
end

% equal weighted average
yhat = mean(yhats,2); 

end