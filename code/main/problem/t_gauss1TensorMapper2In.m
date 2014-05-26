function  t_gauss1TensorMapper2In( op )
%
% - Generate message data set with gentrain_cluttereg.
% - Learn the conditional mean embedding operator
% - Test the operator on a separate test set split from the loaded message 
% samples
% - Measure the error with
%   KL(training Gaussian || operator output Gaussian)
% 
% - This function is for testing Gauss1TensorMapper2In which includes a
% mapper using KMVGauss1 kernel and KNaturalGauss1 kernel. See
% DistMapper2Factory for how to construct these mappers.
%

if nargin < 1
    op = [];
end
op.seed = myProcessOptions(op, 'seed', 1);
seed = op.seed;
rng(seed);

% options specific to this file for choosing which kernel to use
%   @DistMapper2Factory.learnKMVMapper1D for KMVGauss1
%   @DistMapper2Factory.learnKNaturalGaussMapper1D for KNaturalGauss1
op.mapper_learner = myProcessOptions(op, 'mapper_learner', ...
    @DistMapper2Factory.learnKMVMapper1D );

% total samples to use
n = 4000;
ntr = floor(0.8*n);
nte = min(100, n-ntr);

[ s] = l_clutterTrainMsgs( n);
[X, T, Tout, Xte, Tte, Toutte] = Data.splitTrainTest(s, ntr, nte);
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
op.chol_tol = 1e-15;
op.chol_maxrank = min(700, ntr);
op.reglist = [1e-2, 1];

% options used in learning a mapper in DistMapper2Factory
op.med_subsamples = min(1500, ntr);

% Used when selected DistMapper2Factory.learnKMVMapper1D 
op.mean_med_factors = [1];
op.variance_med_factors = [1];

% Used when selected DistMapper2Factory.learnKNaturalGaussMapper1D
op.prec_mean_med_factors = [1]; 
op.neg_prec_med_factors = [1];            
            
% learn a mapper from X to theta
% [mapper, C] = DistMapper2Factory.learnKMVMapper1D(X, T, Tout, op);
% [mapper, C] = DistMapper2Factory.learnKNaturalGaussMapper1D(X, T, Tout, op);
[mapper, C] = feval(op.mapper_learner, X, T, Tout, op);

% new data set for testing EP. Not for learning an operator.
% nN = 50;
% [Theta, tdist] = Clutter.theta_dist(nN);
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
