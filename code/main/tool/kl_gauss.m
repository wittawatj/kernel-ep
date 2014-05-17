function kl = kl_gauss( n1, n2 )
%KL_GAUSS KL divergence between two Gaussians
% 
assert(isa(n1, 'DistNormal'));
assert(isa(n2, 'DistNormal'));
assert(n1.d==n2.d, 'n1 and n2 must have the same dimension');
assert(n1.isproper(), 'n1 is not a proper Gaussian');
assert(n2.isproper(), 'n2 is not a proper Gaussian');

if n1.d==1
    v1 = n1.variance;
    v2 = n2.variance;
    m1 = n1.mean;
    m2 = n2.mean;
    
    kl = ( v1/v2 + ((m1-m2)^2)/v2 -1 -log(v1/v2) )/2;
else
    error('KL for multivariate normal is not supported yet.');
end

end

