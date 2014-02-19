function K = kerLocalScale(X1, X2, k)
%
% Local scaling kernel 
% X1 = d x n1 matrix 
% X2 = d x n2 matrix
% Return K which is a n1 x n2 matrix.
%
% - d is the dimension. 
% - Each instance is a column vector. 
% - X1 and X2 can be just column vectors. 
% - X1 is treated as the training set
% - X2 is not included in the points in which neighbors are calculated.
%
n1 = size(X1,2);
n2 = size(X2,2);
k = k+1;

% pair-wise 2-norm^2 distance of X1 and X1
S11 = sum(X1.^2,1);
D11 = bsxfun(@plus, S11, S11') - 2*(X1'*X1);
V11 = sort(D11,1);
clear D11
% the distance to the kth-neighbor point in X1
% for each instance in X1
kDist1 = V11(k,:);
clear V11

% pair-wise 2-norm^2 distance of X1 and X2
D12 = bsxfun(@plus, sum(X1.^2,1)', sum(X2.^2,1)) - 2*(X1'*X2 );

V12 = sort(D12,1);

% the distance to the kth-neighbor point in X1
% for each instance in X2
kDist2 = V12(k,:);
clear V12

K = exp(-(D12.^2)./( kDist1'*kDist2 )); % kDist1'*kDist2 is an outer product

% K( K <= 1e-50 ) = 0;
% if nnz(K) / numel(K) <= 0.2
% 	K = sparse(K);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end