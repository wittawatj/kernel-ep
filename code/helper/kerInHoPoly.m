function K = kerInHoPoly(X1, X2, d)
%
% Inhomogeneous polynomial kernel. 
%
% X1 = d x n1 matrix 
% X2 = d x n2 matrix
% Return K which is a n1 x n2 matrix.
%
% - d is the dimension. 
% - Each instance is a column vector. 
% - X1 and X2 can be just column vectors. 
%
K = (1+kerDot(X1, X2)).^d;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end