function [ C] = cond_fm_finiteout_cmaes( In, Out, op )
%COND_FM_FINITEOUT_CMAES Generic leave-one-out cross validation procesure for
%selecting FeatureMap candidates for ridge regression with CMA-ES
%   - Use FeatureMap primal solutions (e.g., Rahimi & Recht) 
%   - In is an input Instances objects.
%   - Out must be a matrix where each column represents one instance.
%   - Regularization parameter and all kernel parameters are optimized with 
%   CMA-ES. 
%
assert(nargin>=3, 'struct option (a structure) is mandatory');
% In is likely to be a DistArray which is an Instances
assert(isa(In, 'TensorInstances'));
assert(isnumeric(Out), 'Out must be a matrix');
assert(In.count()==size(Out, 2), 'In and Out must have the same number of instances');

assert(isfield(op, 'featuremap_mode'), 'Field featuremap_mode is required. ');
featuremap_mode = op.featuremap_mode;
% can be joint, sum, product, mv
assert(ismember(featuremap_mode, {'joint', 'sum', 'product', 'mv'}) );

% number of random features to use during LOOCV. 
candidate_primal_features = myProcessOptions(op, 'candidate_primal_features', 500);
assert(candidate_primal_features > 0);

seed = myProcessOptions(op, 'seed', 1);
oldRng = rng;
rng(seed);


if strcmp(featuremap_mode, 'joint')
    fmGenerator = @(params)fm_joint_generator(params, candidate_primal_features);
    [x0, cma_sigma, cma_opt] = fm_joint_initializer(In, Out, candidate_primal_features);

elseif strcmp(featuremap_mode, 'product')
    %nfEach = floor(candidate_primal_features^(1/In.tensorDim()));
    fmGenerator = @(params)fm_product_generator(params, candidate_primal_features);
    [x0, cma_sigma, cma_opt] = fm_product_initializer(In, Out, candidate_primal_features);

elseif strcmp(featuremap_mode, 'mv')
    fmGenerator = @(params)fm_mv_generator(params, candidate_primal_features);
    [x0, cma_sigma, cma_opt] = fm_mv_initializer(In, Out, candidate_primal_features);

else
    error('unknown featuremap_mode: %s', featuremap_mode);
end
% Start cma_es
func = @(jx)(opt_objective(jx, fmGenerator, In, Out));
display(sprintf('Initial point: '));
display(x0);
display(cma_sigma);

[xmin, fmin, counteval, stopflag, cma_out, cma_bestever]= cmaes(func, x0, cma_sigma, cma_opt);

C.minerr = cma_bestever.f;
C.xmin = cma_bestever.x;
C.x0 = x0;
% the first index is for regularization parameter
C.bfeaturemap = fmGenerator(cma_bestever.x(2:end));
C.blambda = cma_bestever.x(1);
C.cma_sigma = cma_sigma;
C.cma_opt = cma_opt;
C.counteval = counteval;
C.stopflag = stopflag;
C.cma_out = cma_out;
C.cma_bestever = cma_bestever;
C.seed = seed;

% set random seed back to the old one
rng(oldRng);
end

function [mse] = opt_objective(jointx, fmGenerator, In, Out)
% Optimize kernel parameters and regularization parameter with CMA-ES.
% - x is a column vector (containing all parameters)
%
% dz x n
if any(jointx <= 0)
    % hard threshold constraint
    mse = nan;
    return;
end

Z = Out;
lambda = jointx(1);
x = jointx(2:end);

fm = fmGenerator(x);
% Subsample each call to the objective function (noisy objective function)
subsample = 1000;
I = randperm(length(In), min(length(In), subsample) );
subIn = In.instances(I);

dm = fm.genFeaturesDynamic(subIn);
[D, n] = size(dm);
PPt = dm.mmt();
subZ = Z(:, I);
C = dm.rmult(subZ')'; % dz x D

A = PPt + lambda*eye(D);
opts.POSDEF = true;
opts.SYM = true;
% this line can be expensive. DxD inverse. O(D^3) complexity.
% D may be large enough so that O(D^3) is expensive. 
T = linsolve(A', C', opts)'; % dz x D
clear A
hdiag = dm.dmtim(lambda, PPt);
% H tilde inverted
HTI =  1./(1- hdiag); % 1xn

B = bsxfun(@times, subZ, HTI); % dz x n
E = dm.lmult(T); % dz x n
EHTI = bsxfun(@times, E, HTI); % dz x n
mse = ( B(:)'*B(:) - 2*EHTI(:)'*B(:) + EHTI(:)'*EHTI(:) )/n;
assert(mse >= 0);

end

function [fm] = fm_product_generator(params, numFeatures)
    % params does not have lambda 
    dim = length(params);
    nfEach = floor(numFeatures^(1/dim));
    fm = RFGProductEProdMap(params, nfEach);
end

function [x0, cma_sigma, cma_opt] = fm_product_initializer(In, Out, numFeatures)
    % nfEach = number of random features for each input
    %
    medf = 1;
    FMs = RFGProductEProdMap.candidatesAvgCov(In, medf, numFeatures );
    fm = FMs{1};

    % let first dimension in x0 be the regularization parameter (lambda)
    lambda = 1e-4;
    params_length = length(fm.gwidth2s);
    x0 = [lambda; fm.gwidth2s(:)];
    

    cma_opt = defaultCMAOpt(length(x0));
    %determines the initial coordinate wise standard deviations for the search
    cma_sigma = [1e-2; 30*fm.gwidth2s(:)];
    cma_sigma = shrinkage(cma_sigma);
end

function [fm] = fm_joint_generator(params, nf)
    % params does not have lambda 
    fm = RFGJointEProdMap(params, nf);
end

function [x0, cma_sigma, cma_opt] = fm_joint_initializer(In, Out, nf)
    % nf = number of random features
    %
    medf = 1;
    FMs = RFGJointEProdMap.candidatesAvgCov(In, medf, nf );
    fm = FMs{1};

    % let first dimension in x0 be the regularization parameter (lambda)
    lambda = 1e-4;
    params_length = length(fm.gwidth2s);
    x0 = [lambda; fm.gwidth2s(:)];

    cma_opt = defaultCMAOpt(length(x0));
    %determines the initial coordinate wise standard deviations for the search
    cma_sigma = [1e-2; 30*fm.gwidth2s(:)];
    cma_sigma = shrinkage(cma_sigma);
end

function [fm] = fm_mv_generator(params, nf)
    % params does not have lambda 
    d = length(params);
    mwidth2s = params(1:(d/2));
    vwidth2s = params( (d/2+1):end);
    fm = RandFourierGaussMVMap(mwidth2s, vwidth2s, nf);
end

function [x0, cma_sigma, cma_opt] = fm_mv_initializer(In, Out, nf)
    % nf = number of random features
    %
    medf = 1;
    FMs = RandFourierGaussMVMap.candidatesFlatMedf(In, medf, nf );
    fm = FMs{1};

    % let first dimension in x0 be the regularization parameter (lambda)
    lambda = 1e-4;
    mwidth2s = fm.mwidth2s;
    vwidth2s = fm.vwidth2s;
    %params_length = length(mwidth2s) + length(vwidth2s);
    x0 = [lambda; mwidth2s(:); vwidth2s(:)];

    cma_opt = defaultCMAOpt(length(x0));
    %determines the initial coordinate wise standard deviations for the search
    cma_sigma = [1e-2; 30*mwidth2s(:); 30*vwidth2s(:)];
    cma_sigma = shrinkage(cma_sigma);
end

function cma_opt = defaultCMAOpt(xlength)

    cma_opt = struct();
    % Stop if the objective function value is below StopFitness
    cma_opt.StopFitness = 1e-8;
    % Each function evaluation involves O(nf^3) (LOOCV with nf dims)
    cma_opt.MaxFunEvals = 100;
    %cma_opt.PopSize = 2*(4+floor(3*log(xlength)));
    cma_opt.MaxIter = 10;
    cma_opt.LBounds = 1e-7;
    %cma_opt.UBounds = 1e6;
    cma_opt.DiffMinChange = [1e-4; 1e-4*ones(xlength-1, 1)];
    cma_opt.TolX = 1e-3;
    cma_opt.TolFun = 1e-4;
    cma_opt.Noise.on = true;
    cma_opt.Restarts = 5;
    %cma_opt.CMA.active = true;
    cma_opt.DispModulo = 1;
    cma_opt.LogModulo = 0;
    cma_opt.SaveVariables = 'off';
end

function sigmas=shrinkage(sigmas)
    % Shrink the search sigma so that the vector is well-conditioned. 
    % max/min is not too large.
    %
    while max(sigmas)/min(sigmas) > 1e3 
        I = sigmas==max(sigmas);
        sigmas(I) = sigmas(I)/1.5;

    end
    
end


