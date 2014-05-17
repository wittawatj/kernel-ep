function K = kerEGaussian1(X1, X2, sigma2)
%
% Expected product Gaussian kernel for mean embeddings of 1d Gaussian
% distributions.
% 
% X1 = 2 x n1 matrix 
% X2 = 2 x n2 matrix
% Return K which is a n1 x n2 matrix where K_ij represents the inner
% product of the mean embeddings of Gaussians in the RKHS induced by the 
% Gaussian kernel with width sigma2. Since the mapped distributions and the
% kernel are Gaussian, the kernel evaluation can be computed analytically.
% 
% Each column must be a 2d vector x with x(1) representing the mean and
% x(2) representing the variance of the distribution. Only 1d Gaussian is
% supported by this function.
% 

d1 = size(X1, 1);
d2 = size(X2, 1);
assert(d1==2, '%s: dimension of X1 must be 2.', mfilename);
assert(d2==2, '%s: dimension of X2 must be 2.', mfilename);
assert(sigma2 > 0, '%s: Gaussian kernel width must be > 0.', mfilename);

M1 = X1(1, :);
M2 = X2(1, :);
V1 = X1(2, :);
V2 = X2(2, :);
% width matrix
W = sigma2 + bsxfun(@plus, V1', V2);

% W(W<=0) = 0.05;
D = bsxfun(@minus, M1', M2).^2;
% normalizer matrix
Z = sqrt(sigma2./W);
% assert(all(imag(Z(:))==0));
% ### hack to prevent negative W in case V1, V2 contain negative variances
if any(imag(Z)>0)
    warning('In %s, kernel matrix contains imaginary entries.', mfilename);
end
Z(imag(Z)~=0) = 0;
K = Z.*exp(-D./(2*W) );



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end