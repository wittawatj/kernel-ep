function W=simmatGaussian(X, gamma)
%
% Construct a similarity or affinity matrix for X with Gaussian kernel.
% Each column of X is one instance. 
%

if exist('pdist','file')
    D = pdist(X');
    W = exp(-(D.^2)./(gamma^2));
    W = squareform(W);
else
    W = exp(-(ipdm(X').^2)./(2*gamma^2));
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end