function W=simmatLocalScale(X, k)
%
% Construct a similarity or affinity matrix for X with Knn+Gaussian kernel
% with local scaling heuristic for the Gaussian width.
% Each column of X is one instance. 
%
k = k+1;

if exist('pdist','file')
    D = squareform(pdist(X'));
else
    D = ipdm(X');
end

[V ] = sort(D,1);
kDist = V(k,:); % the distance to the kth-neighbor for each instance

W = exp(-(D.^2)./( kDist'*kDist )); % kDist'*kDist is an outer product



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
