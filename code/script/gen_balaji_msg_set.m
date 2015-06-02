function [ ] = gen_balaji_msg_set( )
%GEN_BALAJI_MSG_SET This code takes in a collection of EP incoming messages
%collected from Infer.NET on a logistic factor, run an importance sampler to compute 
%outgoing messages, and save the results. Balaji asked for this dataset on 2
%June 2015.
%

% change seed
seed = 1;
oldRng = rng();
rng(seed);

%>> whos
%  Name         Size             Bytes  Class     Attributes

%  Xte1       500x4              16000  double              
%  Xte2       500x4              16000  double              
%  Xtr       4012x4             128384  double              
%  Ytr       4012x1              32096  double              
%
%- Two incoming messages to the factor: Beta(a, b) and N(m, v)
%- The four columns of X are [a, b, m, -log(v)]
%
% load the data set
%>> load('/nfs/nhome/live/wittawat/SHARE/gatsby/research/code/script/logistic_msg/demo_uncertainty_msgs.mat')
data = load('demo_uncertainty_msgs');
% incoming gaussian messages
A_tr = data.Xtr(:, 1)';
B_tr = data.Xtr(:, 2)';
Mean_tr = data.Xtr(:, 3)';
Var_tr = exp(-data.Xtr(:, 4))';

Beta_tr = DistBeta(A_tr, B_tr);
Gauss_tr = DistNormal(Mean_tr, Var_tr);

Beta_te1 = DistBeta(data.Xte1(:, 1)', data.Xte1(:, 2)');
Gauss_te1 = DistNormal(data.Xte1(:, 3)', exp(-data.Xte1(:, 4))' );
Beta_te2 = DistBeta(data.Xte2(:, 1)', data.Xte2(:, 2)');
Gauss_te2 = DistNormal(data.Xte2(:, 3)', exp(-data.Xte2(:, 4))' );


op = struct();

% A forward sampling function taking samples (array) from in_proposal and
% outputting samples from the conditional distribution represented by the
% factor.
op.cond_factor = @(x)1./(1+exp(-x));

op.seed = seed;

% importance sampling data size. 
op.iw_samples = 1e5;

% proposal distribution 
op.in_proposal = DistNormal(0, 100);

% Importance weight vector can be a numerically zero vector when, for
% example, the messages have very small variance. iw_trials specifies the
% number of times to draw IW samples to try before giving up on the
% messages.
op.iw_trials = 2;

% Instead of sampling from the in_proposal, if sample_cond_msg is true,
% then sample from mt (message from T) instead. T is the conditioned
% variable. If true, in_proposal is not needed and ignored.
op.sample_cond_msg = false;

% true to use multicore package for parallel processing. 
op.use_multicore = false;

% p(z|x). The DistBuilder for z (beta). Don't need outgoing Beta.
op.left_distbuilder = [];

% p(z|x). The DistBuilder for x (gaussian).
op.right_distbuilder = DistNormal.getDistBuilder();

[ Beta_tr_new, Gauss_tr_new, ~, Gauss_tr_out ] = ...
    gentrain_dist2(Beta_tr, Gauss_tr, op);

% make sure no example was dropped. This can happen when the importance sampler 
% fails to get a good estimate of the projected message.
assert(length(Beta_tr_new) == length(Beta_tr));

[ Beta_te1_new, Gauss_te1_new, ~, Gauss_te1_out ] = ...
    gentrain_dist2(Beta_te1, Gauss_te1, op);
assert(length(Gauss_te1_new) == length(Gauss_te1));

[ Beta_te2_new, Gauss_te2_new, ~, Gauss_te2_out ] = ...
    gentrain_dist2(Beta_te2, Gauss_te2, op);
assert(length(Gauss_te2_new) == length(Gauss_te2));


% save all the variables
% output label Y = (mean, log precision)
Ytr = [ [Gauss_tr_out.mean] ; -log([Gauss_tr_out.variance]) ]';
Yte1 = [ [Gauss_te1_out.mean]; -log([Gauss_te1_out.variance]) ]';
Yte2 = [ [Gauss_te2_out.mean]; -log([Gauss_te2_out.variance]) ]';
Xtr = data.Xtr;
Xte1 = data.Xte1;
Xte2 = data.Xte2;
particles = op.iw_samples;

fname = sprintf('balaji_msg_set-par%d.mat', particles);
fpath = fullfile(Global.getScriptFolder(), 'logistic_msg', fname); 
timeStamp = clock();

% export all variables to the base workspace.
allvars = who;
warning('off','putvar:overwrite');
putvar(allvars{:});

save(fpath, 'timeStamp', 'Ytr', 'Yte1', 'Yte2', 'Xtr', 'Xte1', 'Xte2', 'particles');
%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(oldRng);
end


