function  t_ggauss_chol_compare(  seed)
%T_GGAUSS_CHOL_COMPARE Compare KGGauss with incomplete Cholesky to
%KGGauss using full matrix

if nargin < 1
    seed = 1;
end
rng(seed);
regen = 0;

fpath = 'saved/test_ggauss_chol_compare.mat';
if exist(fpath, 'file') && ~regen
    load(fpath);
else
    % generate some data
    % options
    n = 3000;
    op.training_size = n;
    op.iw_samples = 2e4;
    op.seed = seed;
    
    % parameters for clutter problem
    a = 10;
    w = 0.5;
    op.clutter_a = a;
    op.clutter_w = w;
    
    % generate training set
    [ X, T, Xout, Tout ] = gentrain_cluttereg(op);
    
    % CV
    % sort dataset by the means of Tout
    [Tout_means, I] = sort([Tout.mean]);
    X = X(I);
    T = T(I);
    Xout = Xout(I);
    Tout = Tout(I);
    
    save(fpath, 'n', 'op', 'a', 'w', 'X', 'T', 'Xout', 'Tout');
    
end

% total samples to use
n = 1100;

% use remove. So we can keep the sorted inputs.
toremove = length(X)-min(n, length(X));
Id = randperm( length(X),  toremove);
X(Id) = [];
T(Id) = [];
Xout(Id) = [];
Tout(Id) = [];

% Learn operator with cross validation
% In = tensor of X and T
% XIns = ArrayInstances(X);
XIns = Gauss1Instances(X);
% TIns = ArrayInstances(T);
TIns = Gauss1Instances(T);
In = TensorInstances({XIns, TIns});

% kernel candidates
embed_widths = [2];
med_factors = [1];

subsamples = n;
SX = XIns.getAll();
ST = TIns.getAll();
KXcell = KGGauss1.candidates(SX, embed_widths, med_factors, subsamples);
KTcell = KGGauss1.candidates(ST, embed_widths, med_factors, subsamples);
% kernel on tensor product space. Stack in the same order as in
% TensorInstances.
Kcandid = KProduct.cross_product(KXcell, KTcell);

op.kernel_candidates = Kcandid;
ToutSuff = [[Tout.mean]; [Tout.variance] + [Tout.mean].^2];

% options for repeated hold-outs
op.num_ho = 1;
op.train_size = floor(0.8*n);
op.test_size = min(1000, n - op.train_size);
op.chol_tol = 1e-15;
op.chol_maxrank = min(500, n);

% learn operator
op.reglist = 1e-3;
[Op, C] = CondCholFiniteOut.learn_operator(In, ToutSuff,  op);

% for full-matrix 
% CV
op.xembed_widths = embed_widths;
op.yembed_widths = embed_widths;
op.xgauss_factors = med_factors;
op.ygauss_factors = med_factors;
op.fold = 2;
Opfull = CondOpGGauss2.learn_operator(X, T, Tout, op);

Isub = 1:ceil(n/100):n;
ncut = length(Isub);

OpMsgs = DistNormal.empty(0, ncut);
OpfullMsgs = DistNormal.empty(0, ncut);
% test the operator on the training set.

for i=1:ncut
    is = Isub(i);
    qni = T(is);
    mxi_f = X(is);
    
    % instance in the tensor product (domain of the operator)
    in = In.instances(is);
    zout = Op.map(in);
    
    % we are working with a 1d Gaussian (for now)
    % mean = zout(1), uncenter 2nd moment = zout(2)
    q = DistNormal(zout(1), zout(2)-zout(1)^2);
    mfi_z_full = Opfull.apply_ep( mxi_f,  qni);

    % Tout, Xout actually contain q not outgoing messages.
%     q = mfi_z*qni;
    qfull = mfi_z_full*qni;
    
    OpMsgs(i) = q;
    OpfullMsgs(i) = qfull;
end

% plot to compare training and output messages
% means
OpfullMeans = [OpfullMsgs.mean];
OpMeans = [OpMsgs.mean];
figure
hold on
set(gca, 'fontsize', 20);
stem(OpfullMeans, 'or');
stem(OpMeans, 'ob');
plot( abs(OpfullMeans-OpMeans), '-k', 'LineWidth', 2);
xlabel('Message index');
ylabel('Gaussian mean');
title(sprintf('ICholesky vs. full kernel matrix. Training size: %d. ichol rank: %d',...
    n, size(Op.R,1) ));
legend('Means (full matrix)', 'Means (ichol)', 'abs. diff.');
grid on
hold off

% variance
OpfullVar = [OpfullMsgs.variance];
OpVar = [OpMsgs.variance];
figure
hold on
set(gca, 'fontsize', 20);
stem(OpfullVar, 'or');
stem(OpVar, 'ob');
plot( abs(OpfullVar-OpVar), '-k', 'LineWidth', 2);
xlabel('Message index');
ylabel('Gaussian variance');
title(sprintf('ICholesky vs. full kernel matrix. Training size: %d. ichol rank: %d', ...
    n, size(Op.R, 1) ));
legend('Variance (full matrix)', 'Variance (ichol)', 'abs. diff.');
grid on
hold off

keyboard


end

