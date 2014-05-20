function  g_clutterTrainMsgs( )
%G_CLUTTERTRAINMSGS Generate training messages for clutter problem.
%
seed = 1;
rng(seed);

fpath = 'saved/clutterTrainMsgs.mat';

% generate some data
% options
n = 1e4;
op.training_size = n;
op.iw_samples = 5e4;
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

