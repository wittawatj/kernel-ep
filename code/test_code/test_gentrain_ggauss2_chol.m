function  test_gentrain_ggauss2_chol( seed )
%
% - Generate message data set with gentrain_cluttereg.
% - Learn the GGauss conditional mean embedding operator
% - Test the operator on the training messages.
% - Measure the error with
%   KL(training Gaussian || operator output Gaussian)
%

if nargin < 1
    seed = 1;
end
rng(seed);
regen = 0;

fpath = 'saved/test_gentrain_ggauss2_chol_data.mat';
if exist(fpath, 'file') && ~regen
    load(fpath);
else
    % generate some data
    % options
    n = 10000;
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
n = 1000;

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
embed_widths = [1, 6, 12];
med_factors = [1/4, 1, 4];

subsamples = max(1000, n);
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
op.num_ho = 3;
op.train_size = floor(0.8*n);
op.test_size = min(1000, n - op.train_size);
op.chol_tol = 1e-5;
op.reglist = [1e-4, 1e-2, 1];

% max rows in R for K ~ R'*R
op.chol_maxrank = min(500, n);
% learn operator
[Op, C] = CondCholFiniteOut.learn_operator(In, ToutSuff,  op);

% new data set for testing EP. Not for learning an operator.
% nN = 50;
% [Theta, tdist] = theta_dist(nN);
% [NX, xdist] = ClutterMinka.x_cond_dist(Theta, a, w);


Isub = 1:ceil(n/100):n;
ncut = length(Isub);
KL = zeros(1, ncut);
OpMsgs = DistNormal.empty(0, ncut);
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
%     mfi_z = Op.apply_ep( mxi_f,  qni);

    % Tout, Xout actually contain q not outgoing messages.
    
    true_toTq = Tout(is);
    if true_toTq.isproper() && q.isproper()
        % compare mfi_z to the one from training set
        kl = kl_gauss(true_toTq, q);
    else
        kl = nan();
    end
    KL(i) = kl;
    OpMsgs(i) = q;
end

% plot to compare training and output messages
% means
TrMeans = [Tout(Isub).mean];
OpMeans = [OpMsgs.mean];
figure
hold on
set(gca, 'fontsize', 20);
stem(TrMeans, 'or');
stem(OpMeans, 'ob');
plot( abs(TrMeans-OpMeans), '-k', 'LineWidth', 2);
xlabel('Message index');
ylabel('Gaussian mean');
title(sprintf('Means. Training size: %d', n));
legend('Train means', 'Output means', 'abs. diff.');
grid on
hold off

% variance
TrVar = [Tout(Isub).variance];
OpVar = [OpMsgs.variance];
figure
hold on
set(gca, 'fontsize', 20);
stem(TrVar, 'or');
stem(OpVar, 'ob');
plot( abs(TrVar-OpVar), '-k', 'LineWidth', 2);
xlabel('Message index');
ylabel('Gaussian variance');
title(sprintf('Variances. Training size: %d', n));
legend('Train variance', 'Output variance', 'abs. diff.');
grid on
hold off

% plot KL
figure
stem(KL);
set(gca, 'fontsize', 20);
title('KL error on training messages' );
xlabel('Messsage index');
ylabel('KL');


keyboard

end

