function K = kerEGaussian(X1, X2, sigma2)
%
% Expected product Gaussian kernel for mean embeddings of Gaussian
% distributions. Equivalently, compute the inner product of mean embedding
% using Gaussian kernel.
%
% Return K which is a n1 x n2 matrix where K_ij represents the inner
% product of the mean embeddings of Gaussians in the RKHS induced by the
% Gaussian kernel with width sigma2. Since the mapped distributions and the
% kernel are Gaussian, the kernel evaluation can be computed analytically.
%
% X1, X2 = 1xn array of DistNormal.
%

% [~, n1] = size(X1);
% [~, n2] = size(X2);
assert(isa(X1, 'DistNormal'));
assert(isa(X2, 'DistNormal'));

if sigma2 <= 0
    error('%s: Gaussian kernel width must be > 0.', mfilename);
end

dx1 = X1(1).d;
dx2 = X2(1).d;

if dx1 == 1 && dx2 == 1
    % mean's of X
    MX1 = [X1.mean];
    % variance's of X
    VX1 = [X1.variance];
    MX2 = [X2.mean];
    VX2 = [X2.variance];
    PX1 = [MX1; VX1];
    PX2 = [MX2; VX2];
    
    K = kerEGaussian1(PX1, PX2, sigma2);
else
    error('Multi-variate Gaussians are not support yet.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end