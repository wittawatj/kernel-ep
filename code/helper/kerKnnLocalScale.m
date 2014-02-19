function K = kerKnnLocalScale(X1, X2, k)
%
% Local scaling kernel + Knn. 
% X1 = d x n1 matrix 
% X2 = d x n2 matrix
% Return K which is a n1 x n2 matrix.
%
% - d is the dimension. 
% - Each instance is a column vector. 
% - X1 and X2 can be just column vectors. 
% - X1 is treated as the training set
%
error('not finished implementing');


n1 = size(X1,2);
n2 = size(X2,2);

% pair-wise 2-norm^2 distance of X1 and X2
D = bsxfun(@plus, sum(X1.^2,1)', sum(X2.^2,1)) - 2*(X1'*X2 );


[V I] = sort(D,1);
Ik = I(1:k,:); should be k+1 
I2 = Ik + n*(0:(n-1)) ;
if k < n/3 % if k is low enough, make a sparse W
    II = reshape(Ik, k*n,1);
    JJ = reshape(repmat(1:n, k, 1), k*n, 1);
    Ds = sparse(II, JJ, D(I2), n1, n2);
else
    
end


kDist = V(k,:); % the distance to the kth-neighbor for each instance
W = exp(-(D.^2)./( kDist'*kDist )); % kDist'*kDist is an outer product



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end