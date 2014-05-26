% function t_condop1( seed)
%
% Test conditional mean embedding C_z|x trained on data generated from
% p(x | z)p(z) which is not the correct p(z|x)p(x)
%
% if nargin < 1
%     seed = 1;
% end
seed = 2;
oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);

% parameters for clutter problem
a = 10;
w = 0.5;

N  = 2000;
% from
fr = -7;
to = 7;
train_theta = unifrnd(fr, to, 1, N);
tp_pdf = @(t)unifpdf(t, fr, to);
% train_theta = randn(1, N)*sqrt(10) + 3;

[train_X, xdist, feval] = ClutterMinka.x_cond_dist(train_theta, a, w);
% learn operator

% operator at f_i from x to theta. C_theta|x
options = [];
options.reglist  = [ 1e-4,  1e-2, 1, 10];
options.xwlist = [1/8, 1/5, 1, 2, 4, 8];
options.fold = 2;
% options.xwlist = [1];
[Op] = CondOp1.learn_operator(train_X, train_theta, options);

%%%%%%%% testing

ntest = 80;
% [Theta] = 3*ones(1, ntest) + 0.1*randn(1, ntest);
[Theta] = 3*ones(1, ntest) + 0.1*randn(1, ntest);
% draw new X for testing conditioning on the Theta
test_X = ClutterMinka.x_cond_dist(Theta, a, w);

% proposal for theta in importance sampling
s = DistNormal(3, 10);
draw = 1e4;
Tpro = s.draw(draw);
Tden = s.density(Tpro);
Spro = DistNormal.suffStat(Tpro);
% records
OpMes = DistNormal.empty(); % operator messages
IwMes = DistNormal.empty(); % importance-weight messages

Op_q = DistNormal(0, Inf);
Iw_q = DistNormal(0, Inf);

for i=1:size(test_X, 2)
    x = test_X(:, i);
    % we observed X. Use PointMass.
    %     mxi_f = PointMass(x);
    % we observed X. But, let's put a width around it
    mxi_f = DistNormal(x, 0.1);
    mfi_t = Op.apply_bp(mxi_f);
    Op_q = Op_q * mfi_t;
    OpMes(i) = mfi_t;
    
    % importance sampling
    W = feval( repmat(x, 1, length(Tpro)), Tpro )./ Tden;
    % projection
        suffStat = Spro*W'/draw;
%     suffStat = Spro*W'/sum(W);
    me = suffStat(1);
    cov = suffStat(2) - me*me' ;
    % importance weighted message
    iw_mfi_t = DistNormal(me, cov);
    Iw_q = Iw_q * iw_mfi_t;
    IwMes(i) = iw_mfi_t;
end

% analyze

% means
Op_means = [OpMes.mean];
Op_vars = [OpMes.variance];
Iw_means = [IwMes.mean];
Iw_vars = [IwMes.variance];

hold all
set(gca, 'fontsize', 20);
stem(Op_means, '-r', 'linewidth', 2);
stem(Iw_means, '-b', 'linewidth', 2);
plot(abs(Op_means - Iw_means), '-k', 'linewidth', 2)
legend('Means (operator)', 'Means (importance)', '|Op-Iw|');
title(sprintf('BP messages. $q_{op} = \\mathcal{N}(%.3g, %.3g), q_{iw} = \\mathcal{N}(%.3g, %.3g)$', ...
    Op_q.mean, Op_q.variance ,Iw_q.mean, Iw_q.variance), 'interpreter', 'latex');
xlabel('$f_i$', 'interpreter', 'latex');
grid on
hold off

% variance
figure
hold all
set(gca, 'fontsize', 20);
stem(Op_vars, '-r', 'linewidth', 2);
stem(Iw_vars, '-b', 'linewidth', 2);
plot(abs(Op_vars- Iw_vars), '-k', 'linewidth', 2)
legend('variance (operator)', 'variance (importance)', '|Op-Iw|');
title(sprintf('BP messages. $q_{op} = \\mathcal{N}(%.3g, %.3g), q_{iw} = \\mathcal{N}(%.3g, %.3g)$', ...
    Op_q.mean, Op_q.variance ,Iw_q.mean, Iw_q.variance), 'interpreter', 'latex');

xlabel('$f_i$', 'interpreter', 'latex');
grid on
hold off

%%%%%%%%%%%%%%%%%%%%%%%
RandStream.setGlobalStream(oldRs);


% keyboard
% end

