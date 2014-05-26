% solve clutter problem  with original EP
seed = 18;
oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);

a = 10;
w= 0.5;

C = ClutterMinka(a, w);

N = 100;
% Theta = 4*ones(1, N);

% make sure to keep it the same as it clutter_problem.m
if false
    Theta = randn(1, N) + 3;
    [X, fx] = ClutterMinka.x_cond_dist(Theta, a, w);
else
    load('main/problem/clutter/clutter_data.mat');
    X = observedData;
end

% observedData = X;
% save('main/problem/clutter/clutter_data.mat', 'observedData', 'Theta');

% initial values for q
m0 = 0;
v0 = 10;
% TM (iterations x N) = mean of each i for each iteration t
[R] = C.ep(X, m0, v0, seed );


% fplot(@(x)pdf(fx, x), [-5, 5])

RandStream.setGlobalStream(oldRs);

%%%%%%%%%%%%%%%
plot_epfacs(R);