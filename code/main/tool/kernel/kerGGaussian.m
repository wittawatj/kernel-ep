function [K, D2] = kerGGaussian(X1, X2, sigma2, w2)
%
% Gaussian kernel of mean embeddings with parameter w2. The distributions
% in X1, X2 are also embedded with a Gaussian kernel with parameter sigma2.
%
% Return: K which is a n1 x n2 matrix.
% Input: X1, X2 = 1xn array of DistNormal.
%
if w2 <= 0
    error('%s: Gaussian kernel width w2 must be > 0.', mfilename);
end

[ D2] = distGGaussian( X1, X2, sigma2);
K = exp( -D2/(2*w2) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
