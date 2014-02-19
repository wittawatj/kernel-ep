function K = kerGaussian(X1, X2, sigma2)
%
% Gaussian kernel. 
% X1 = d x n1 matrix 
% X2 = d x n2 matrix
% Return K which is a n1 x n2 matrix.
%
% - d is the dimension. 
% - Each instance is a column vector. 
% - X1 and X2 can be just column vectors. 
%
% n1 = size(X1,2);
% n2 = size(X2,2);

D2 = bsxfun(@plus, sum(X1.^2,1)', sum(X2.^2,1)) - 2*(X1'*X2 );
% D2 = repmat(sum(X1'.^2,2), 1, n2) ...
%     + repmat(sum(X2.^2,1), n1, 1) ...
%     - 2*X1'*X2;

K = exp(-D2./(2*(sigma2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end