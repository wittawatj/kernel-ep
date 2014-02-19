function W=simmatKnn(X, k)
%
% Construct a similarity or affinity matrix for X with Knn.
% Each column of X is one instance. The number of neighbours to 
% consider can exceed k since we want the notion of neighbor to be
% symmetric. If x1 is neighbor to x2, then x2 will be set to neighbor to x1
% as well.
%
k=k+1;

D = ipdm(X');
n = size(D,1);
[V I] = sort(D,1);
I = I(1:k, :);


W = [];
if k < n/3 % if k is low enough, make a sparse W
    J = repmat(1:n, k, 1);
    W = sparse(I(:), J(:), true, n, n );
else
    I2 = I + repmat( n*(0:(n-1)) ,k,1);
    W = false(n,n);
    W(I2) = true;
end

% Make it symmetric with 'or' 
W = or(W,W');

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end