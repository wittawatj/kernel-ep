function [R, op] = herd_weights_l1(bundle, op)
% HERD_WEIGHTS_L1 Learn weights on location sample points for sending EP outgoing 
% messages by finding the weights combination of sufficient statistics at the 
% location points.
%   - bundle = a MsgBundle
%

assert(isstruct(op));
assert(isa(bundle, 'MsgBundle'));

% required inputs
% This out_msg_distbuilder must correspond to the variable identified by 
% target_index.
out_msg_distbuilder = op.out_msg_distbuilder;
% An instance of CondFactor
cond_factor = op.cond_factor;
assert(isa(cond_factor, 'CondFactor'));

% index of the variable to be treated as the message sending target.
% f(x_1 | x_2, x_3, ....)
target_index = op.target_index;
assert(target_index >= 1);

% -- optional inputs ---

% random seed 
op.seed = myProcessOptions(op, 'seed', 1);
seed = op.seed;
oldRng = rng();
rng(seed);
% maximum number of conditioning locations to consider. 
% This corresponds to the maximum number of non-zero coefficients in Lasso.
op.max_locations = myProcessOptions(op, 'max_locations', 5e3);
max_locations = op.max_locations;

% number of lambda (regularization parameter in Lasso) to consider.
op.num_lambda = myProcessOptions(op, 'num_lambda', 50);
assert(op.num_lambda > 0);
num_lambda = op.num_lambda;

% #folds on cross validation
op.cv_fold = myProcessOptions(op, 'cv_fold', 3);
cv_fold = op.cv_fold;

op.lambdas = myProcessOptions(op, 'lambdas', []);
lambdas = op.lambdas;

% conditioning points for the factors.
% Either this options or max_locations must be specified.
assert(isfield(op, 'cond_points') || isfield(op, 'max_locations'), ...
    'At least one of {cond_points, max_locations} must be specified.');
if isfield(op, 'cond_points')
    cond_points = op.cond_points;
    assert(isa(cond_points, 'MatTensorInstances'));
else
    % cond_points not specified. Require max_locations.
    % max_locations = number of location points to draw 
    max_locations = op.max_locations;
    assert(max_locations > 0);
    % Draw a quasi Monte Carlo sequence to uniformly conver the space.
    cond_points = cond_factor.batchDrawQuasiMCPoints(max_locations);
end
outLoc = cond_factor.sample(cond_points);
[X, TCell] = getX(bundle, target_index, outLoc, cond_points, out_msg_distbuilder );

Result = struct();
Lasso = struct();
% get Y
inDaCell = bundle.getInputBundles();
targetDa = inDaCell{target_index};
nTarget = out_msg_distbuilder.getStat(targetDa);
tau = size(nTarget, 1);
Gamma = kron(eye(max_locations), ones(1, tau));

% consider each output separately
for j=1:tau
    Yj = nTarget(j, :);
    % DFmax = maximum number of non-zero coefficients
    % Rely on Matlab's lasso()
    [B, FitInfo] = lasso(X'*Gamma', Yj', 'DFmax', max_locations, ...
        'NumLambda', num_lambda, 'RelTol', 1e-2, 'Standardize', false, ...
        'CV', cv_fold, 'Lambda', lambdas);

    Lasso(j).B = B;
    Lasso(j).FitInfo = FitInfo;
end

Result.Lasso = Lasso;
Result.nTarget = nTarget;
Result.locationSuffCell = TCell;
Result.cond_points = cond_points;
Result.outLocation = outLoc;
R = Result;

rng(oldRng)
end

function [X, TCell] = getX(bundle, target_index, outLoc, cond_points, out_msg_distbuilder )
    % Return the X matrix in the herding L1 problem (data matrix of Lasso 
    % problem).
    assert(isnumeric(outLoc));
    assert(isa(cond_points, 'MatTensorInstances'));
    condPointsCell = cond_points.matsCell;
    % getInputBundles() returns a cell array which contains output as well.
    inDaCell = bundle.getInputBundles();
    if target_index == 1
        targetLoc = outLoc;
        % DistArray of the target variable
    else
        targetLoc = condPointsCell{target_index-1};
    end

    % Important: Need to make sure that transformStat() and getStat() are 
    % consistent with each other.
    TCell = out_msg_distbuilder.transformStat(targetLoc);
    %OutT = out_msg_distbuilder.getStat(outDa);
    tranStat = vertcat(TCell{:});
    tau = size(tranStat, 1);
    %assert(size(OutT, 1) == tau);
    K = length(cond_points);
    ntr = length(bundle);
    X = zeros(tau*K, ntr);

    pointsCell = [{outLoc}, condPointsCell] ;
    for j=1:ntr
        % 1 x n
        Den = 1.0;
        for i=1:length(inDaCell)
            celli = inDaCell{i};
            mij = celli(j);
            pointsi = pointsCell{i};
            mijDen = mij.density(pointsi);
            Den = Den.*mijDen;
        end
        Tw = bsxfun(@times, Den, tranStat);
        X(:, j) = Tw(:);
    end
end % end getX()

