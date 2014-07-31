function [ C] = cond_fm_finiteout( In, Out, op )
%COND_FM_FINITEOUT Generic leave-one-out cross validation procesure for
%selecting FeatureMap candidates 
% for conditional mean embedding
%   - Learn C_{Out|In}
%   - The conditional mean embedding can take any number of inputs. This
%   simply depends on the Kernel used (kernel on tensor product space).
%   - Use FeatureMap primal solutions (e.g., Rahimi & Recht) 
%   - In is an input Instances objects.
%   - Out must be a matrix where each column represents one instance.
%
assert(nargin>=3, 'op (a structure) is mandatory');
% In is likely to be a DistArray which is an Instances
assert(isa(In, 'Instances'));
assert(isnumeric(Out), 'Out must be a matrix');
assert(In.count()==size(Out, 2), 'In and Out must have the same number of instances');

% a cell array of FeatureMap candidates to try.
assert(isfield(op, 'featuremap_candidates'), ...
    'Field featuremap_candidates is require in op. A cell array of FeatureMap.');
featuremap_candidates = op.featuremap_candidates;
assert(all(cellfun(@(k)(isa(k, 'FeatureMap')) , featuremap_candidates) ));

% list of regularization parameters.
reglist = myProcessOptions(op, 'reglist', [1e-2, 1e-0, 10]);
op.reglist = reglist;

% Boolean to use or not use multicore package
gop=globalOptions();
use_multicore = myProcessOptions(op, 'use_multicore', gop.use_multicore);

% multicore settings. A structure.
multicore_settings = myProcessOptions(op, 'multicore_settings', struct());

seed = myProcessOptions(op, 'seed', 1);
oldRng = rng;
rng(seed);

fmEvalFunc = @(fm)(evalFeatureMap(fm, In, Out, op));    
if use_multicore
    % call multicore package. Evaluate one FeatureMap with one slave.
    % resultCell = cell of HR
    gop=globalOptions();
    multicore_settings.multicoreDir= myProcessOptions(multicore_settings, ...
        'multicoreDir', gop.multicoreDir);
    resultCell = startmulticoremaster(fmEvalFunc, featuremap_candidates, multicore_settings);
    
    CErr = [resultCell{:}];
else
    % case: Not use multicore
    % Matrix of regression square loss. num_ho x reg x candidates
    CErr = inf(length(reglist), length(featuremap_candidates) );
    for fi=1:length(featuremap_candidates)
        fm = featuremap_candidates{fi};
        R = fmEvalFunc(fm);
        CErr(:, fi) = R;
    end
end

% best param combination
[minerr, ind] = min(CErr(:));
[bri, bfi] = ind2sub(size(CErr), ind);

% At this point, we have bri, bfi
C.minerr = minerr;
C.bfeaturemap= featuremap_candidates{bfi};
C.bfeaturemap_index=bfi;
C.blambda = reglist(bri);

C.reglist = reglist;
C.featuremap_candidates= featuremap_candidates;
C.seed = seed;

% set random seed back to the old one
rng(oldRng);
end

function [R] = evalFeatureMap(fm, In, Out, op)
% Evaluate a FeatureMap fm 
% Return a column R vector containing errors for each regularization parameter. 
%
% dz x n
Z = Out;
% Incomplete Cholesky of the best kernel candidate
assert(isa(fm, 'FeatureMap'));

reglist = op.reglist;
R = inf(length(reglist), 1);
% Phi hat matrix. Call it P (D x n) where D is the number of primal features.
% Fixed for all regularization parameters.
%
% !! Not a good idea to explicitly form P.

%P = fm.genFeatures(In);
%[D, n] = size(P);
%PPt = P*P';
%C = Z*P';    

% Do the above lines with DynamicMatrix
dm = fm.genFeaturesDynamic(In);
[D, n] = size(dm);
PPt = dm.mmt();
C = dm.rmult(Z')'; % dz x D

for ri=1:length(reglist)
    lambda = reglist(ri);
    
    A = PPt + lambda*eye(D);
    opts.POSDEF = true;
    opts.SYM = true;
    % this line can be expensive. DxD inverse. O(D^3) complexity.
    % D may be large enough so that O(D^3) is expensive. 
    % But, this is certainly better than O(N^3) where dual solution is used.
    % *** Should use conjugate gradient here ***
    % *** Improve later *** 
    T = linsolve(A', C')'; % dz x D
    clear A
    hdiag = dm.dmtim(lambda, PPt);
    % H tilde inverted
    HTI =  1./(1- hdiag); % 1xn

    B = bsxfun(@times, Z, HTI); % dz x n
    E = dm.lmult(T); % dz x n
    EHTI = bsxfun(@times, E, HTI); % dz x n
    mse = ( B(:)'*B(:) - 2*EHTI(:)'*B(:) + EHTI(:)'*EHTI(:) )/n;

    %T = bsxfun(@times, B, HTI)*E'; %dz x dz
    %M = B*B' - T - T' - bsxfun(@times, E, HTI.^2)*E';
    %mse = M(:)'*M(:)/n;
    R(ri) = mse;

    fprintf('loocv: lamb: %.3g, fm: %s => err: %.3g\n', ...
        lambda, fm.shortSummary(), mse);
end


end

