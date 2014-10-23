function [ C] = cond_ho_finiteout_cmaes( In, Out, op )
%COND_HO_FINITEOUT_CMAES Generic repeated hold-out procesure for selecting kernel
%parameters for ridge regression with CMA-ES
%   - Learn C_{Out|In}
%   - Use incomplete Cholesky on the kernel matrix. Factorization option
%   can be specified in op (structure).
%   - In is an input Instances objects.
%   - Out must be a matrix where each column represents one instance.
%
assert(nargin>=3, 'op (a structure) is mandatory');
assert(isa(In, 'Instances'));
assert(isnumeric(Out), 'Out must be a matrix');
assert(In.count()==size(Out, 2), 'In and Out must have the same number of instances');

n = size(Out, 2);

kernel_mode = op.kernel_mode;
% can be kggauss, ...
assert(ismember(kernel_mode, {'kggauss'}));

% Number of hold-outs :=h to perform. Train h times on different randomly
% drawn h sets and test on a seperate nonoverlapping test set.
num_ho = myProcessOptions(op, 'num_ho', 3 );
assert(num_ho>=1, 'num_ho (#hold-outs) must be at least 1');
op.num_ho = num_ho;

% Training set size. Training and test sets are randomly drawn and non-overlapping.
% Default = 80%
ho_train_size = myProcessOptions(op, 'ho_train_size', floor(0.8*n));
op.ho_train_size = ho_train_size;

% Test set size. Train and test sizes do not have to add up to n.
% Default = 20%
ho_test_size = myProcessOptions(op, 'ho_test_size',  n-ho_train_size );
op.ho_test_size = ho_test_size;

% Tolerance for incomplete Cholesky on kernel matrix
chol_tol = myProcessOptions(op, 'chol_tol', 1e-8);
op.chol_tol = chol_tol;

% Maximum rank (#rows of R) K~R'*R in incomplete Cholesky for training
chol_maxrank_train = myProcessOptions(op, 'chol_maxrank_train', max(n, 300) );
op.chol_maxrank_train = chol_maxrank_train;

% Maximum rank (#rows of R) K~R'*R in incomplete Cholesky after model selection
chol_maxrank = myProcessOptions(op, 'chol_maxrank', max(n, 600) );
op.chol_maxrank = chol_maxrank;

% seed for data stratification
seed = myProcessOptions(op, 'seed', 1);
oldRng = rng;
rng(seed);

if strcmp(kernel_mode, 'kggauss')
    kerGenerator = @(params)ker_kggauss_generator(params);
    [x0, cma_sigma, cma_opt] = ker_kggauss_initializer(In, Out);
else
    error('unknown kernel_mode: %s', kernel_mode);
end

% Start cma_es 
func = @(jx)eval_kernel(jx, kerGenerator, In, Out, op);
[xmin, fmin, counteval, stopflag, cma_out, cma_bestever]= cmaes(func, x0, cma_sigma, cma_opt);

C.minerr = cma_bestever.f;
C.xmin = cma_bestever.x;
%C.minerr = fmin;
%C.xmin = xmin;

C.x0 = x0;
% the first index is for regularization parameter
C.bkernel = kerGenerator(C.xmin(2:end));
% Do Cholesky again. We could have stored but that will disable parfor ?
BestKerIChol = IncompChol(In, C.bkernel, chol_tol, chol_maxrank);
C.bkernel_ichol = BestKerIChol;
C.blambda = C.xmin(1);

C.num_ho = num_ho;
C.ho_train_size = ho_train_size;
C.ho_test_size = ho_test_size;
C.chol_tol = chol_tol;
C.chol_maxrank_train = chol_maxrank_train;
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

function [mse] = eval_kernel(jointx, kerGenerator, In, Out, op)
% Evaluate the kernel function kfunc.
% Return HR matrix, a error matrix of size num_ho x reg_list
%

% expand all vars in op here
eval(structvars(100, op));

Z = Out;
% Incomplete Cholesky of the best kernel candidate

lambda = jointx(1);
x = jointx(2:end);
kfunc = kerGenerator(x);
assert(isa(kfunc, 'Kernel'));

% Instances indices
n = size(Z, 2);
HR = inf(num_ho, 1);
% Sample only ho_train_size for training and test on ho_test_size samples.
% This function is stochastic.
[TRI, TEI] = train_test_indices(num_ho, ho_train_size, ho_test_size, n );
for hoi=1:num_ho

    % It is user's responsibility to make sure the kernel can support
    % instances in In.   
    ichol = IncompChol(In.instances(TRI(hoi, :)), kfunc, chol_tol, chol_maxrank_train);

    % Make test index
    teI = TEI(hoi, :);
    % Make training index
    trI = TRI(hoi, :);
    %         ntr = length(trI);
    nte = length(teI);
    Intr = In.get(trI);
    Inte = In.get(teI);
    
    % reduced Cholesky matrix on training set
    %Rr = ichol.R(:, trI);
    Rr = ichol.R;
    Krs = kfunc.eval(Intr, Inte); %tr x te
    RrKrs = Rr*Krs;
    Ztr = Z(:, trI);
    Zte = Z(:, teI);
    t1 = Zte(:)'*Zte(:)/nte;
    
    B = (Rr*Rr' + lambda*eye(size(Rr,1))) \ RrKrs;
    Q = (Ztr*Krs - (Ztr*Rr')*B )/lambda; % dz x te
    t2 = Q(:)'*Q(:)/nte;
    t3 = -2*Zte(:)'*Q(:)/nte;

    mse = t1+t2+t3;
    HR(hoi) = mse;

    %fprintf('random hold-out: lamb: %.3g, ker: %s => err: %.3g\n', ...
    %    lambda, kfunc.shortSummary(), mse);
end
mse = mean(HR);

end

function ker = ker_kggauss_generator(params)
    % !!! dimension length should be fixed for multivariate case !!!
    % Assume univariate case for now.
    %
    % Assume one KGGaussian takes 2 parameters
    assert(mod(length(params), 2)==0);
    numKers = length(params)/2;
    kerCell = cell(1, numKers);
    j = 1;
    for i=1:2:length(params)
        kerCell{j} = KGGaussian(params(i), params(i+1));
        j = j+1;
    end
    ker = KProduct(kerCell);
end

function [x0, cma_sigma, cma_opt] = ker_kggauss_initializer(In, Out)

    %numVars = In.tensorDim();
    medf = 1;
    kers = KGGaussian.productCandidatesAvgCov(In, medf, 2000);
    ker = kers{1};
    assert(isa(ker, 'KProduct'));
    params = kproductKGGauss2Params(ker);

    % let first dimension in x0 be the regularization parameter (lambda)
    lambda = 1e-4;
    x0 = [lambda; params(:)];

    cma_opt = defaultCMAOpt(length(x0));
    %determines the initial coordinate wise standard deviations for the search
    cma_sigma = [1e-2; 30*params(:)];
    cma_sigma = shrinkage(cma_sigma);
end

function params = kproductKGGauss2Params(kpro)
    % Get all parameters of KProduct of KGGaussian's
    cellParams = kpro.getParam();
    params = [cellParams{:}];
    params = [params{:}];
end

function cma_opt = defaultCMAOpt(xlength)

    cma_opt = struct();
    % Stop if the objective function value is below StopFitness
    cma_opt.StopFitness = 1e-8;
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
        sigmas(I) = sigmas(I)*0.9;

    end
    
end

function [TRI, TEI] = train_test_indices(num_ho, tr, te, n)
% tr = training size, te = test size
% TRI = num_ho x tr (subset of 1:n)
% TEI = num_ho x te = subset of 1:n but disjoint from TRI
assert(tr+te<=n, 'train + test sizes must not exceed n');

TRI = zeros(num_ho, tr);
TEI = zeros(num_ho, te);
for i=1:num_ho
    I = randperm(n);
    TRI(i, :) = I(1:tr);
    TEI(i, :) = I( (tr+1):(tr+te));
end

end


