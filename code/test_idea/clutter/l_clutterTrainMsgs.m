function [ s] = l_clutterTrainMsgs( nload )
%L_CLUTTERTRAINMSGS Load training messages for clutter problem.
% n = number of instances to load
% 

fpath = 'saved/clutterTrainMsgs.mat';
assert(exist(fpath, 'file')~=0 );
load(fpath);
% load() will load 'n', 'op', 'a', 'w', 'X', 'T', 'Xout', 'Tout'

% use remove. So we can keep the sorted inputs.
toremove = length(X)-min(nload, length(X));
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

% pack everything into a struct s
s.op = op;
s.a = a;
s.w = w;
s.X = X;
s.T = T;
s.Xout = Xout;
s.Tout = Tout;

s.XIns = XIns;
s.TIns = TIns;
s.In = In;

end

