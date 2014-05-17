function [op] = learn_cond_operator1(X, Z_X)
% learn mean embedding operator taking 1 input
% Gaussian kernels. Do cross validation.
% The operator can be thought of a mapping of messages x->f
% into f->z where the factor f is represented by the conditional sample of
% Z_X on X
% 

n = size(X,2);
o.fold = 2;
C = cond_embed_cv1(X, Z_X, o);

% medx = pairwise median distance of X
skx = C.bxw * C.medx; %bxw = best Gaussian width for x

lamb = C.blambda;
% Kernel matrix
Kx = kerGaussian(X, X, skx);

% inv is bad. But improve later.
O = inv(Kx + lamb*eye(n));

op.O = O;

op.data = {X, Z_X};
op.in = 1;
op.params = {skx};
end
