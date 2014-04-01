% solve clutter problem  with original EP
seed = 16;
oldRs = RandStream.getGlobalStream();
            rs = RandStream.create('mt19937ar','seed',seed);
            RandStream.setGlobalStream(rs);
            
            
a = 10;
w= 0.5;

C = ClutterMinka(a, w);

N = 200;
% Theta = 4*ones(1, N);
Theta = randn(1, N) + 3;

% [X, ftrue] = ClutterMinka.x_cond_dist(Theta, a, w);

% initial values for q
m0 = 0;
v0 = 10;
% TM (iterations x N) = mean of each i for each iteration t
[R] = C.ep(Theta, m0, v0, seed );


% fplot(@(x)pdf(fx, x), [-5, 5])

RandStream.setGlobalStream(oldRs);
