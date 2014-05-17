function three_gaussians()
% x,y,z on one factor in the factor graph.
% Everything is a Gaussian to test the idea first.

% Parameters of messages from x,y,z to factor
mx = 0;
sx = 1;

my = 1;
sy = 2;
% priors Z (m_z_to_factor) not conditional Z|X,Y
mz = 2;
sz = 1;

% Generate samples from message x and y
n = 400;
X = randn(1, n)*sqrt(sx) + mx;
Y = randn(1, n)*sqrt(sy) + my;
a = 1;
b = -2;
Z_XY = randn(1, n) + a*X+ b*Y  ;

% Training steps.
%%%%%
% to z
opz = learn_cond_operator2(X, Y, Z_XY);

% % to x. Can't do this. Can't use Z_XY as Z. We need a prior sample for Z
% not conditional sample.
opx = learn_cond_operator2(Y, Z_XY, X);

opy = learn_cond_operator2(X, Z_XY, Y);

% EP message passing
%%%%%%%%
xs = [mx, sx];
ys = [my, sy];
zs = [mz, sz];

% suff z
display(sprintf('Expected: meanz: %.3f, varz: %.3f', a*mx+b*my, a*a*sx+b*b*sy+1));
tzs = normal_suff_stat2(xs, ys, zs, opz)

% suff x
display(sprintf('Expected: mx: %.3f, varx: %.3f', mx, sx));
txs = normal_suff_stat2(ys, zs, xs, opx)
tys = normal_suff_stat2(xs, zs, ys, opy)

% EP iterations
% T = 50;
% for t=1:T
%     
%     % do for each target node: x, y, z
%     tzs = normal_suff_stat(xs, ys, zs, opz);
%     txs = normal_suff_stat(ys, zs, xs, opx);
%     tys = normal_suff_stat(xs, zs, ys, opy);
% end
warning('three_gaussians example is deprecated because division after projection is not needed? ');

% keyboard
end


