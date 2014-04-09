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
    load('test_idea/clutter_data.mat');
    X = observedData;
end

% observedData = X;
% save('test_idea/clutter_data.mat', 'observedData', 'Theta');

% initial values for q
m0 = 0;
v0 = 10;
% TM (iterations x N) = mean of each i for each iteration t
[R] = C.ep(X, m0, v0, seed );


% fplot(@(x)pdf(fx, x), [-5, 5])

RandStream.setGlobalStream(oldRs);

%%%%%%%%%%%%%%%
% Plot results mean
figure
hold all
stem(R.TM(1, :) , '-b', 'linewidth', 1);
stem(R.TM(2, :) , '-r', 'linewidth', 1);
plot(R.TMQNI(1, :), '-k', 'linewidth', 2);
plot(R.TMQNI(2, :), '-g', 'linewidth', 2);
% plot(R.TM(1, :) , 'linewidth', 1, 'markerfacecolor', 'blue');
% plot(R.TM(2, :) , 'linewidth', 1, 'markerfacecolor', 'red');

set(gca, 'fontsize', 20);
xlabel('Factor index');
ylabel('Value');
title(sprintf('Mean of $\\tilde{f}_i$. q: (%.3g, %.3g) ', R.m, R.v), 'Interpreter', 'latex');
legend('Iteration 1', 'Iteration 2', 'It. 1: q^{\\i}', 'It. 2: q^{\\i}');
ylim([-10, 10])
grid on
hold off

% TV variance
figure
hold all
stem(R.TV(1, :) , '-b', 'linewidth', 1);
stem(R.TV(2, :) , '-r', 'linewidth', 1);
plot(R.TVQNI(1, :), '-k', 'linewidth', 2);
plot(R.TVQNI(2, :), '-g', 'linewidth', 2);
% plot(R.TM(1, :) , 'linewidth', 1, 'markerfacecolor', 'blue');
% plot(R.TM(2, :) , 'linewidth', 1, 'markerfacecolor', 'red');

set(gca, 'fontsize', 20);
xlabel('Factor index');
ylabel('Value');
title(sprintf('Variance of $\\tilde{f}_i$. q: (%.3g, %.3g) ', R.m, R.v), 'Interpreter', 'latex');
legend('Iteration 1', 'Iteration 2', ...
    'It. 1: q^{\\i}', 'It. 2: q^{\\i}');
ylim([-30, 30])
grid on
hold off
