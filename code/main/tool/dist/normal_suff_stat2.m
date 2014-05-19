function [tzs] = normal_suff_stat2(xs, ys, zs, op)
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
% If we is observed e.g., x=1, then use the following instead of taking an
% expectation.
% Mux = kerGaussian(X, 1, z_skx); % 

Muy = sqrt(2*pi*z_sky)*normpdf(Y(:), my, sqrt(sy+z_sky));
% Muy = kerGaussian(Y, -2, z_sky);

% mu hat of z. \sum_i=1^n \alpha_i \phi(z_i) 
Alpha = Oz*( Mux.*Muy);
Beta = normpdf(Z_XY(:), mz, sqrt(sz));
C = Alpha.*Beta; 
% C = Alpha; 
meanz = Z_XY*C;
% varz = Z_XY.^2*C - meanz^2;
% tzs = [meanz, varz];

% with division by incoming msg z->f after the projection
nvarz= (varz^-1 - sz^-1)^-1;
tzs = [nvarz*((varz^-1)*meanz - (sz^-1)*mz) , nvarz];

end