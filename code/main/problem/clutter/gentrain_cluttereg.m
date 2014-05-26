function [ X, T, Xout, Tout ] = gentrain_cluttereg( op)
%GENTRAIN_CLUTTEREG Generate training set (messages) for clutter problem.
% 
% T = theta
% Tout = outgoing messages for theta (after projection)
% 

if nargin < 1
    op = [];
end

N = myProcessOptions(op, 'training_size', 1000);

% gen X, gen T
X = gen_x_msg(N);
T = gen_theta_msg(N);

% proposal distribution for for the conditional varibles (i.e. t) 
% in the factor. Require: Sampler & Density.
op.in_proposal = DistNormal( 0, 30);

% clutter problem specific parameters
a = myProcessOptions(op, 'clutter_a', 10);
w = myProcessOptions(op, 'clutter_w', 0.5);

% A forward sampling function taking samples (array) from in_proposal and
% outputting samples from the conditional distribution represented by the
% factor.
op.cond_factor = @(T)(ClutterMinka.x_cond_dist(T, a, w));
[ X, T, Xout, Tout ] = gentrain_dist2(X, T, op);
 
end


function X=gen_x_msg(n)
% 
M = randn(1, n)*sqrt(30);
% Should focus on low variance because we will observe X and represent it
% with a DistNormal having a small variance (instead of using a PointMass).
V = gamrnd(1, 0.1, 1, n);
% V = unifrnd(0.01, 1, 1, n);
X = DistNormal(M, V);
end


function T=gen_theta_msg(n)
% MT = unifrnd(-10, 10, 1, n);
MT = randn(1, n)*sqrt(30);
% VT = gamrnd(2, 100, 1, n);
VT = unifrnd(0.01, 1000, 1, n);
T = DistNormal(MT, VT);
end

