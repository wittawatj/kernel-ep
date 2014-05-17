function med = medEGaussian( X )
%MEDEGAUSSIAN Pairwise median distance heuristic for Gaussian message
%dataset
% 
% X is an array of DistNormal. Compute the pairwise median distance
% heuristic by finding the median of {mu_i*Sigma_i^-1}.
% 

if X(1).d > 1
    error('Multivariate Gaussians are not supported yet.');
end

warning('medEGaussian heuristic does not have a good justification.');
M = [X.mean];
V = [X.variance];
% width matrix
W = bsxfun(@plus, V', V);
D2 = bsxfun(@minus, M', M).^2;
D = D2./W;
med = median(D(:));

end

