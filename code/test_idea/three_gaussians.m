function three_gaussians()
% x,y,z on one factor in the factor graph. 
% Everything is a Gaussian to test the idea first.

% Parameters of messages from x,y,z to factor
mx = 1;
sx = 1;

my = 0;
sy = 1;
% priors Z (m_z_to_factor) not conditional Z|X,Y
mz = 1;
sz = 2;

% Generate samples from message x and y
n = 800;
X = randn(1, n)*sqrt(sx) + mx;
Y = randn(1, n)*sqrt(sy) + my;

Z_XY = randn(1, n)*sqrt(1) + X+ Y -1 ;

% Training steps. Learn 3 mappings
%%%%%
% to z
opz = learn_op(X, Y, Z_XY);
% to x
opx = learn_op(Y, Z_XY, X);
% to y
opy = learn_op(X, Z_XY, Y);

% EP message passing
%%%%%%%%
xs = [mx,sx];
ys = [my, sy];
zs = [mz, sz];
T = 50;
% for t=1:T

% for each factor (we have only one factor here)
% do for each target node: x, y, z

% suff z
[tzs] = suff_stat(xs, ys, zs, opz);

% suff y
[tys] = suff_stat(xs, zs, ys, opy);


% suff x
[txs] = suff_stat(xs, zs, ys, opx);
% end

% muh_z = 
keyboard
end

function [tzs] = suff_stat(xs, ys, zs, op)
% compute sufficient statistic of an outgoing message to a node z.
% inputs:
%   - sufficient statistics of all incoming messages to the factor.
%   - kernel parameters associated with operator Oz
%   - Z_XY sample from p(z|x,y)
% return: tzd = suff stat of message to z
Oz=op.O;
z_skx = op.params{1};
z_sky = op.params{2};
X = op.data{1};
Y = op.data{2};
Z_XY = op.data{3};
 
mx = xs(1);
sx = xs(2);
my = ys(1);
sy = ys(2);
mz = zs(1);
sz = zs(2);
% Embedded Mu of x and y
Mux = sqrt(2*pi*z_skx)*normpdf(X(:), mx, sqrt(sx+z_skx));
Muy = sqrt(2*pi*z_sky)*normpdf(Y(:), my, sqrt(sy+z_sky));

% mu hat of z. \sum_i=1^n \alpha_i \phi(z_i) 
Alpha = Oz*( Mux.*Muy);
Beta = normpdf(Z_XY(:), mz, sqrt(sz));
C = Alpha.*Beta;
% C = Alpha;
meanz = Z_XY*C;
varz = Z_XY.^2*C + meanz^2;
tzs = [meanz, varz];

% with division by incoming msg z->f after the projection
% nvarz= (varz^-1 - sz^-1)^-1;
% tzs = [nvarz*((varz^-1)*meanz - (sz^-1)*mz) , nvarz];

end

function [op] = learn_op(X, Y, Z)
% learn mean embedding operator
% Gaussian kernels
% supposed to be doing cross validation here. Need Z sample for CV
n = size(X,2);
% determined by CV
skx = meddistance(X);
sky = meddistance(Y);
% regularization param
lamb = 1e-5;

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
