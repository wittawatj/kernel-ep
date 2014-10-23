function [ D2] = distGGaussian( X1, X2, sigma2)
%DISTGGAUSSIAN Distance^2 matrix before taking the exponential.
% Used in kerGGaussian(). The actual formula is for Gaussian distributions. 
% If X, Y are not Gaussian, treat them as one by moment matching.
%
n1 = length(X1);
n2 = length(X2);
%[~, n1] = size(X1);
%[~, n2] = size(X2);
assert(isa(X1, 'Distribution'));
assert(isa(X2, 'Distribution'));

if X1(1).d ==1
    % operation on obj array can be expensive in Matlab ...
    M1 = [X1.mean];
    M2 = [X2.mean];
    V1 = [X1.variance];
    V2 = [X2.variance];
    
    %M1 = M1(:)';
    %M2 = M2(:)';
    %V1 = V1(:)';
    %V2 = V2(:)';
    T1 = repmat(KEGauss1.self_inner1d(M1, V1, sigma2)', 1, n2);
    T2 = repmat(KEGauss1.self_inner1d(M2, V2, sigma2), n1, 1);
    
    S1 = [M1; V1];
    S2 = [M2; V2];
    Cross = kerEGaussian1(S1, S2, sigma2);
    
    D2 = T1 -2*Cross + T2 ;
    
else
    error('later for multivariate case');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
