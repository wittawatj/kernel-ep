function  g_clutterTrainMsgs(n , op )
%G_CLUTTERTRAINMSGS Generate training messages for clutter problem.
%
if nargin < 2
    op=[];
end

seed = myProcessOptions(op, 'seed', 1);
rng(seed);

clutter_data_path = myProcessOptions(op, 'clutter_data_path', ...
    'saved/clutterTrainMsgs.mat');


% generate some data
% options
op.training_size = n;
op.iw_samples = 7e4;
op.seed = seed;

% sample from msgs from Theta instead of from the proposal
% op.sample_cond_msg = true;

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

save(clutter_data_path, 'n', 'op', 'a', 'w', 'X', 'T', 'Xout', 'Tout');

end

