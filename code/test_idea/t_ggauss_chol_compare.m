function  t_ggauss_chol_compare(  seed)
%T_GGAUSS_CHOL_COMPARE Compare KGGauss with incomplete Cholesky to
%KGGauss using full matrix

if nargin < 1
    seed = 1;
end
rng(seed);


% total samples to use
n= 8000;
[ s] = l_clutterTrainMsgs( n);
% This will load a bunch of variables in s into the current scope.
eval(structvars(100, s));

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
op.num_ho = 3;
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
    zout = Op.mapInstances(in);
    
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

