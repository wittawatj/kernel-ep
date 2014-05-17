function  test_gentrain_egauss2( seed )
%
% - Generate message data set with gentrain_dist2.
% - Learn the Egauss conditional mean embedding operator
% - Test the operator on the training messages.
% - Measure the error with
%   KL(training Gaussian || operator output Gaussian)
%
if nargin < 1
    seed = 1;
end

oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed', seed);
RandStream.setGlobalStream(rs);

% options
op.training_size = 500;
op.iw_samples = 2e4;
op.fold = 2;

% parameters for clutter problem
a = 10;
w = 0.5;
op.clutter_a = a;
op.clutter_w = w;

% CV
op.xwlist = [ 1, 8, 16, 32];
op.ywlist = [ 1, 8, 16, 32];
op.reglist = [1e-4, 1e-2, 1];

% generate training set
[ X, T, Xout, Tout ] = gentrain_cluttereg(op);
% sort dataset by the means of Tout
[Tout_means, I] = sort([Tout.mean]);
X = X(I);
T = T(I);
Xout = Xout(I);
Tout = Tout(I);

% Learn operator with cross validation
Op = CondOpEGauss2.learn_operator(X, T, Tout, op);

% op.xembed_widths = [1, 8, 16];
% op.yembed_widths = [1, 8, 16];
% op.xgauss_factors = [1/4, 1, 4];
% op.ygauss_factors = [1/4, 1, 4];
% Op = CondOpGGauss2.learn_operator(X, T, Tout, op);

% new data set for testing EP. Not for learning an operator.
% nN = 50;
% [Theta, tdist] = theta_dist(nN);
% [NX, xdist] = ClutterMinka.x_cond_dist(Theta, a, w);

n = length(X);
Isub = 1:ceil(n/100):n;
ncut = length(Isub);
KL = zeros(1, ncut);
OpMsgs = DistNormal.empty(0, ncut);
% test the operator on the training set.

for i=1:ncut
    is = Isub(i);
    qni = T(is);
    mxi_f = X(is);
    mfi_z = Op.apply_ep( mxi_f,  qni);
    % Tout, Xout actually contain q not outgoing messages.
    q = mfi_z*qni;
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
RandStream.setGlobalStream(oldRs);
end
