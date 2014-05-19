function  t_kmvmapper1d2in( seed )
%
% - Generate message data set with gentrain_cluttereg.
% - Learn the conditional mean embedding operator
% - Test the operator on the training messages.
% - Measure the error with
%   KL(training Gaussian || operator output Gaussian)
%

if nargin < 1
    seed = 1;
end
rng(seed);

% total samples to use
n = 3000;
ntr = floor(0.8*n);
nte = min(100, n-ntr);

[ s] = l_clutterTrainMsgs( n);
[X, T, Tout, Xte, Tte, Toutte] = splitTrainTest(s, ntr, nte);
assert(length(X)==ntr);
assert(length(T)==ntr);
assert(length(Tout)==ntr);

% This will load a bunch of variables in s into the current scope.
% load 'n', 'op', 'a', 'w', 'X', 'T', 'Xout', 'Tout'
% eval(structvars(100, s));

% options for repeated hold-outs
op.num_ho = 3;
op.train_size = floor(0.7*ntr);
op.test_size = min(1000, ntr - op.train_size);
op.chol_tol = 1e-5;
op.chol_maxrank = min(500, ntr);
op.reglist = [1];

% options used in learnMapper
op.med_subsamples = min(1500, ntr);
op.mean_med_factors = [1/3, 1];
op.variance_med_factors = [1/3, 1];

% learn a mapper from X to theta
[mapper, C] = KMVMapper1D2In.learnMapper(X, T, Tout, op);

% new data set for testing EP. Not for learning an operator.
% nN = 50;
% [Theta, tdist] = theta_dist(nN);
% [NX, xdist] = ClutterMinka.x_cond_dist(Theta, a, w);

% testing
[hmean, hvar, hkl]=distMapper2_gauss1_tester( mapper,  Xte, Tte, Toutte);

% cheating by testing on training set
% limit = 100;
% I = randperm(length(X), min(limit, length(X)) );
% [hmean, hvar, hkl]=distMapper2_gauss1_tester( mapper,  X(I), T(I), Tout(I));


axmean = get(hmean, 'CurrentAxes');
axvar = get(hvar, 'CurrentAxes');
axkl = get(hkl, 'CurrentAxes');
title(axmean, sprintf('Training size: %d', ntr));
title(axvar, sprintf('Training size: %d', ntr));
title(axkl, sprintf('Training size: %d', ntr));
keyboard
end

function [X, T, Tout, Xte, Tte, Toutte] = splitTrainTest(s, ntr, nte)
eval(structvars(100, s));

% respect ntr first
toremove = length(X)-min(ntr, length(X));
Id = randperm( length(X),  toremove);
Xte = X(Id);
X(Id) = [];

Tte = T(Id);
T(Id) = [];

Toutte = Tout(Id);
Tout(Id) = [];

% further reduce to nte if nte < n-ntr
nte = min(length(Xte), nte);
Id2 = randperm(length(Xte), nte);
Xte = Xte(Id2);
Tte = Tte(Id2);
Toutte = Toutte(Id2);
end
