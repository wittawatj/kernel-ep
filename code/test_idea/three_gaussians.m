function three_gaussians()
% x,y,z on one factor in the factor graph. 
% Everything is a Gaussian to test the idea first.

% Parameters of messages from x,y,z to factor
mx = 0;
sx = 1;

my = 1;
sy = 2;
% priors Z (m_z_to_factor) not conditional Z|X,Y
mz = 1;
sz = 1;

% Generate samples from message x and y
n = 400;
X = randn(1, n)*sqrt(sx) + mx;
Y = randn(1, n)*sqrt(sy) + my;
a = 1;
b = -2;
Z_XY = randn(1, n) + a*X+ b*Y  ;

% Training steps. Learn 3 mappings
%%%%%
% to z
opz = learn_cond_operator(X, Y, Z_XY);
% % to x
% opx = learn_op(Y, Z_XY, X);
% % to y
% opy = learn_op(X, Z_XY, Y);

% EP message passing
%%%%%%%%
xs = [mx, sx];
ys = [my, sy];
zs = [mz, sz];
T = 50;
% for t=1:T

% for each factor (we have only one factor here)
% do for each target node: x, y, z

% suff z
[tzs] = normal_suff_stat(xs, ys, zs, opz);


% muh_z = 
keyboard
end


