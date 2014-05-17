function [op] = learn_cond_operator2(X, Y, Z_XY)
% learn mean embedding operator taking 2 inputs
% Gaussian kernels. Do cross validation.
% The operator can be thought of a mapping of messages x->f, y->f
% into f->z where the factor f is represented by the conditional sample of
% Z_XY on X,Y.
% 

n = size(X,2);
o.fold = 2;
C = cond_embed_cv2(X, Y, Z_XY, o);

% medx = pairwise median distance of X
skx = C.bxw * C.medx; %bxw = best Gaussian width for x
sky = C.byw * C.medy;
lamb = C.blambda;
% Kernel matrix
Kx = kerGaussian(X, X, skx);
Ky = kerGaussian(Y, Y, sky);
% inv is bad. But improve later.
O = inv(Kx.*Ky + lamb*eye(n));

op.O = O;

op.data = {X,Y, Z_XY};
op.in = 2;
op.params = {skx, sky};
end
