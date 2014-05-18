% function t_condop1_simple( seed)
%
% Test conditional mean embedding C_x|z trained on data generated from
% the correct p(x|z)p(z)
%
% if nargin < 1
%     seed = 1;
% end

seed = 2;
oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);

% training set
ntrain = 1000;
a = 10;
b = 1/4;
Ztr = gamrnd(a, b, 1, ntrain);
% Ztr = max(0, randn(1, ntrain)*2 + 2.5);
% Ztr = chi2rnd(4, 1, ntrain);
Xtr = exprnd(Ztr);

% operator at f_i from z to x. C_x|z
options = [];
options.reglist  = [ 1e-4,  1e-2, 1];
options.xwlist = [1/5, 1, 2, 4, 10];
options.fold = 2;
% options.xwlist = [1];
[Op] = CondOp1.learn_operator(Ztr, Xtr, options);

%%%%%%%% testing
ntest = 80;
Zte = gamrnd(a, b, 1, ntest);
Xte = exprnd(Zte);

% histogram of X
figure
XX = exprnd(gamrnd(a, b, 1, 2e4));
[counts, xout] = hist(XX, 30);
hold all
bar(xout, counts/sum(counts));
g =@(z)pdf('gamma', z, a, b);
dom = 0:0.01:20;
plot(dom, g(dom), '-r', 'linewidth', 2);

set(gca, 'fontsize', 20);
legend('X marginal (test)' , 'Z (test)');

xlabel('X');
title(sprintf('Marginal X: mean: %.3g, variance: %.3g', mean(XX,2), var(XX) ));
grid on
hold off

% records
OpMes = DistNormal.empty();
Op_q = DistNormal(0, Inf);

for i=1:size(Zte, 2)
    z = Zte(:, i);
    % we observed Z. Use PointMass.
%     mzi_f = PointMass(z);
    % we observed Z. But, let's put a width around it
    mzi_f = DistNormal(z, 0.1);
    mfi_x = Op.apply_bp(mzi_f);
    Op_q = Op_q * mfi_x;
    OpMes(i) = mfi_x;
    
end

% means
Op_means = [OpMes.mean];
Op_vars = [OpMes.variance];

figure
hold all
set(gca, 'fontsize', 20);
stem(Op_vars, '-b',  'linewidth', 2);
stem(Op_means, '-r', 'linewidth', 2);

legend( 'variance', 'Means (operator)');
title(sprintf('BP messages. $q_{op} = \\mathcal{N}(%.3g, %.3g)$. E(means)=%.3g, E(var)=%.3g', ...
    Op_q.mean, Op_q.variance, mean(Op_means), mean(Op_vars) ), 'interpreter', 'latex');
xlabel('$f_i$', 'interpreter', 'latex');
grid on
hold off

% variance
% figure
% hold all
% set(gca, 'fontsize', 20);
% stem(Op_vars, '-r', 'linewidth', 2);
% 
% legend('variance (operator)');
% title(sprintf('BP messages. $q_{op} = \\mathcal{N}(%.3g, %.3g)$', ...
%     Op_q.mean, Op_q.variance), 'interpreter', 'latex');
% 
% xlabel('$f_i$', 'interpreter', 'latex');
% grid on
% hold off

%%%%%%%%%%%%%%%%%%%%%%%
RandStream.setGlobalStream(oldRs);


% keyboard
% end

