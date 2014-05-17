function A=discrete_rnd(P, m, n)
% 
% Draw samples from a discrete distribution with parameter P (Kx1).
% 
if abs( sum(P) - 1) > 1e-8
    warning('Argument to discrete_rnd() does not sum to 1. Something may go wrong.');
end
K = length(P);
P=P(:);
C = cumsum(P)';
I = bsxfun( @lt, rand(m*n,1), C);
nums = K+1-sum(I, 2);
A = reshape(nums, m, n);
end
