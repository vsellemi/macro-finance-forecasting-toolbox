function S = estimate_CSR(Xtr,Ytr,K)

all_ind = nchoosek(1:size(Xtr,2), K);
betas   = NaN(size(all_ind,1),K); 
for c = 1:size(all_ind,1)
    tmpidx = all_ind(c,:); 
    tmpS   = OLS(Xtr(:,tmpidx), Ytr); 
    betas(c,:) = tmpS.Beta;
end

S = struct('idx',all_ind, 'betas',betas); 

end