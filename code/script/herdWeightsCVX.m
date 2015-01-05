%function  Result = herdWeightsCVX(  )
%   .

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
%bunName='sigmoid_bw_proposal_2000';
bunName='sigmoid_bw_proposal_5000';
%bunName='sigmoid_bw_proposal_1000';
%bunName='sigmoid_bw_proposal_10000';
%bunName='sigmoid_bw_proposal_20000';

%bunName='sigmoid_bw_fixbeta_10000';
%bunName='sigmoid_bw_fixbeta_10000';
bundle=se.loadBundle(bunName);

%[trBundle, teBundle] = bundle.partitionTrainTest(800, 200);
[trBundle, teBundle] = bundle.partitionTrainTest(1000, 2000);
%[trBundle, teBundle] = bundle.partitionTrainTest(1000, 1000);
%[trBundle, teBundle] = bundle.partitionTrainTest(16000, 2000);

%Xtr = trBundle.getInputTensorInstances();
%Ytr = out_msg_distbuilder.getStat(trBundle.getOutBundle());
%out_msg_distbuilder = DNormalLogM2Builder();
out_msg_distbuilder = DistNormalBuilder();

% number of location points
K = 300;
% draw uniformly 
Xloc = linspace(-16, 16, K);
%Xloc = normrnd(0, 5, 1, K);
condDistFactor = @(x) 1./(1+exp(-x));
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
Xs = cell(1, tau);
%X = zeros(K, ntr);

for t=1:tau
    X = zeros(K, ntr);
    for i=1:ntr
        % 1 x n
        mz = daz(i);
        mx = dax(i);
        zden = mz.density(Zloc);
        xden = mx.density(Xloc);
        Txw = bsxfun(@times, zden.*xden, Tx(t, :));
        X(:, i) = Txw(:);
    end
    Xs{t} = X;
end
%Gamma = kron(eye(K), ones(1, tau));

l1_bound = 1e-0;
l2_bound = 1e-1;

%l1_reg = 1e0;
%l2_reg = 1e1;
Result = struct();

B = zeros(K, tau);
%Intercept = zeros(1, tau);
% consider each output separately
for j=1:tau
    Yj = OutTx(j, :);
    %Xj = [Xs{j}; ones(1, ntr)];
    Xj = [Xs{j}];
    % DFmax = maximum number of non-zero coefficients
    %
    cvx_begin
      cvx_precision medium
      variable be(K)
      %minimize ( quad_form( Xj'*be - Yj', eye(ntr) ));
      minimize( norm( Xj' * be - Yj', 2 )/ntr )
      subject to
          %ones(1, K) * be <= l1_bound 
          norm(be, 1) <= l1_bound 
          norm(be, 2) <= l2_bound
          %be'*be <= l2_bound^2
          be >= 0
    cvx_end

    %cvx_begin
    %  cvx_precision medium
    %  variable be(K)
    %  minimize( norm( Xs{j}' * be - Yj', 1 )/ntr + l1_reg*ones(1, K)*be/K + l2_reg*norm(be, 2)/K )
    %  subject to
    %      be >= 0
    %cvx_end
    B(:, j) = be(1:K);
    clear be;
end

cond_points = MatTensorInstances({Xloc});
instancesMappers = cell(1, tau);
for j=1:tau
    weights = B(:, j)';
    intercept = 0;

    instancesMappers{j} = HerdInstancesMapper(weights, intercept, ...
        Tx(j, :),  Zloc, cond_points);
end

im = StackInstancesMapper(instancesMappers{:});
dm=GenericMapper(im, out_msg_distbuilder, bundle.numInVars());

tester = DivDistMapperTester(dm);

% save
n=length(trBundle)+length(teBundle);
ntr = length(trBundle);
iden=sprintf('herdCVX_%s_ntr%d_K%d.mat', bunName, ntr, K);
fpath=Expr.scriptSavedFile(iden);

timeStamp=clock();
save(fpath, 'dm', 'B', 'OutTx', 'timeStamp', 'trBundle', 'teBundle');
rng(oldRng);
%end


