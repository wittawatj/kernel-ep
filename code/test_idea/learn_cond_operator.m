
function [op] = learn_cond_operator(X, Y, Z)
% learn mean embedding operator
% Gaussian kernels. Do cross validation.

n = size(X,2);
o.fold = 2;
C = cond_embed_cv2(X, Y, Z, o);

skx = C.bxw * C.medx;
sky = C.byw * C.medy;
lamb = C.blambda;
% Kernel matrix
Kx = kerGaussian(X, X, skx);
Ky = kerGaussian(Y, Y, sky);
% inv is bad. But improve later.
O = inv(Kx.*Ky + lamb*eye(n));

op.O = O;

op.data = {X,Y, Z};
op.in = 2;
op.params = {skx, sky};
end