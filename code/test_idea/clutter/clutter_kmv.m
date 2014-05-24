function [R] = clutter_kmv( op )
% - Clutter problem of Tom Minka solved by learning a conditional mean
% embedding operator with KMVGauss1 kernel
% - Specifically, use KMVMapper1D2In
%

if nargin < 1
    op=[];
end

op.seed = myProcessOptions(op, 'seed', 1);
seed = op.seed;
rng(seed);

op.clutter_model_path = myProcessOptions(op, 'clutter_model_path', ...
    'saved/clutter_kmv_mapper.mat');
% Boolean flag
op.retrain_clutter_model = myProcessOptions(op, 'retrain_clutter_model', false);
if ~exist(op.clutter_model_path, 'file') || op.retrain_clutter_model
    
    % total samples to use
    op.total_samples = myProcessOptions(op, 'total_samples', 8000);
    n = op.total_samples;
    ntr = floor(0.8*n);
    nte = min(100, n-ntr);
    
    % op.clutter_data_path = 'saved/clutterTrainMsgs_mgauss_vgam.mat';
    % op.clutter_data_path = 'saved/clutterTrainMsgs.mat';
    op.clutter_data_path = 'saved/clutterTrainMsgs_noproposal.mat';
    
    % Load training messages
    [ s] = l_clutterTrainMsgs( n, op);
    % load 'n', 'op', 'a', 'w', 'X', 'T', 'Xout', 'Tout'
%     eval(structvars(100, s));
    % load variables in s
    a = s.a;
    w = s.w;
   
    [X, T, Tout, Xte, Tte, Toutte] = Data.splitTrainTest(s, ntr, nte);
    assert(length(X)==ntr);
    assert(length(T)==ntr);
    assert(length(Tout)==ntr);
    
    % options for repeated hold-outs
    op.num_ho = 4;
    op.train_size = floor(0.7*ntr);
    op.test_size = min(2000, ntr - op.train_size);
    op.chol_tol = 1e-5;
    op.chol_maxrank = min(700, ntr);
    op.reglist = [1e-2, 1];
    % op.use_multicore = false;
    
    % options used in learnMapper
    op.med_subsamples = min(1500, ntr);
    op.mean_med_factors = [1];
    op.variance_med_factors = [1];
    
    % options for EP (for distMapper2_ep() )
    op.f0 = DistNormal(0, 30);
    op.ep_iters = 10;
    op.observed_variance = 0.1;
    op.mean_conv_thresh = 0.05;
    op.var_conv_thresh = 0.5;
    
    % learn a mapper from X to theta
    [mapper, C] = KMVMapper1D2In.learnMapper(X, T, Tout, op);
    save(op.clutter_model_path, 'mapper', 'C', 'saved_op', 'a', 'w', 's');
else
    % mapper already exists in a file. load it.
    load(op.clutter_model_path);
    op = dealstruct(saved_op, op);
end

% new data set for testing EP. Not for learning an operator.
op.observed_size = myProcessOptions(op, 'observed_size', 500);
nN = op.observed_size;
% the mean of theta
op.clutter_theta_mean = myProcessOptions(op, 'clutter_theta_mean', 3);

% [Theta, tdist] = Clutter.theta_dist(nN);
Theta = randn(1, nN)*sqrt(0.1) + op.clutter_theta_mean ;
[NX, xdist] = ClutterMinka.x_cond_dist(Theta, a, w);

% start EP
[ R] = distMapper2_ep( NX, mapper, op);

% records
op.data = s;
R.op = op;
% % Minka's EP
% C = ClutterMinka(a, w);
% % initial values for q
% m0 = 0;
% v0 = 10;
% % TM (iterations x N) = mean of each i for each iteration t
% [R] = C.ep(NX, m0, v0, seed);
%
% % Clutter problem solved with importance sampling projection
% theta_proposal = DistNormal(2, 20);
% iw_samples = 2e4;
% IC = ClutterImportance(a, w, theta_proposal, iw_samples);
% IR =  IC.ep( NX, m0, v0, seed);

%%% plot
% plot_clutter_results(tdist, xdist, tp_pdf, q, seed);
% kernel EP
% [h1,h2]= plot_epfacs( KR );
% [h1,h2]= plot_epfacs( R );

% keyboard
%%%%%%%%%%%%%%%%%%%%%%%%%%5
end


