function  Result = herdWeightsPrototype(  )
%HERDWEIGHTSPROTOTYPE Herding weights
%   .


se=BundleSerializer();
%bunName='sigmoid_bw_proposal_5000';
bunName='sigmoid_bw_proposal_1000';
%bunName='sigmoid_bw_proposal_10000';
%bunName='sigmoid_bw_proposal_20000';

%bunName='sigmoid_bw_fixbeta_10000';
%bunName='sigmoid_bw_fixbeta_10000';
bundle=se.loadBundle(bunName);

[trBundle, teBundle] = bundle.partitionTrainTest(200, 400);
%[trBundle, teBundle] = bundle.partitionTrainTest(16000, 2000);

%Xtr = trBundle.getInputTensorInstances();
%Ytr = out_msg_distbuilder.getStat(trBundle.getOutBundle());
%out_msg_distbuilder = DNormalLogVarBuilder();
out_msg_distbuilder = DistNormalBuilder();

% number of location points
K = 300;
% draw uniformly 
Xloc = linspace(-16, 16, K);
Zloc = condDistFactor(Xloc);
ntr = length(trBundle);
daz = trBundle.getInputBundle(1);
% Use DistBetaArray for efficiency of density().
daz = DistBetaArray(daz);
dax = trBundle.getInputBundle(2);
dax = DistNormalArray(dax);

% Important: Need to make sure that transformStat() and getStat() are 
% consistent with each other.
TxCell = out_msg_distbuilder.transformStat(Xloc);
% pick X to be the target in f(z|x)
targetDa = trBundle.getInputBundle(2);
OutTx = out_msg_distbuilder.getStat(targetDa);
Tx = vertcat(TxCell{:});
tau = size(Tx, 1);
assert(size(OutTx, 1) == tau);
% Sufficient statistic of x 
% length of sufficient statisitc
X = zeros(tau*K, ntr);

for i=1:ntr
    % 1 x n
    mz = daz(i);
    mx = dax(i);
    zden = mz.density(Zloc);
    xden = mx.density(Xloc);
    Txw = bsxfun(@times, zden.*xden, Tx);
    X(:, i) = Txw(:);
end
Gamma = kron(eye(K), ones(1, tau));

Result = struct();
lambdas = [1e-2];
% consider each output separately
for j=1:tau
    Yj = OutTx(j, :);
    % DFmax = maximum number of non-zero coefficients
    [B, FitInfo] = lasso(X'*Gamma', Yj', 'DFmax', 200, 'NumLambda', 20, ...
        'RelTol', 1e-3, 'Standardize', false, 'CV', 2, 'Lambda', lambdas );

    Result(j).B = B;
    Result(j).FitInfo = FitInfo;
end

end


function z = condDistFactor(x)
    % sigmoid
    z = 1./(1+exp(-x));
end


