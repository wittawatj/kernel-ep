function [h1,h2]= plot_epfacs( R )
% Plot the results of running EP. The results include the mean and variance
% of each f tilde. 
% 

%%%%%%%%%%%%%%%

% Plot results mean
figure
hold all
stem(R.TM(1, :) , '-b', 'linewidth', 1);
stem(R.TM(end, :) , '-r', 'linewidth', 1);
plot(R.TMQNI(1, :), '-k', 'linewidth', 2);
plot(R.TMQNI(end, :), '-g', 'linewidth', 2);
% plot(R.TM(1, :) , 'linewidth', 1, 'markerfacecolor', 'blue');
% plot(R.TM(2, :) , 'linewidth', 1, 'markerfacecolor', 'red');
h1 = gca;
set(gca, 'fontsize', 20);
xlabel('Factor index');
ylabel('Value');
title(sprintf('Mean of $\\tilde{f}_i$. q: (%.3g, %.3g). Totally %d iter. ', ...
    R.m, R.v, size(R.TM,1) ), 'Interpreter', 'latex');
legend('It. 1', 'It. last', 'It. 1: q^{\\i}', 'It. last: q^{\\i}');
% ylim([-10, 10])
grid on
hold off

% TV variance
figure
hold all
stem(R.TV(1, :) , '-b', 'linewidth', 1);
stem(R.TV(end, :) , '-r', 'linewidth', 1);
plot(R.TVQNI(1, :), '-k', 'linewidth', 2);
plot(R.TVQNI(end, :), '-g', 'linewidth', 2);
% plot(R.TM(1, :) , 'linewidth', 1, 'markerfacecolor', 'blue');
% plot(R.TM(2, :) , 'linewidth', 1, 'markerfacecolor', 'red');
h2 = gca;
set(gca, 'fontsize', 20);
xlabel('Factor index');
ylabel('Value');
title(sprintf('Variance of $\\tilde{f}_i$. q: (%.3g, %.3g). Totally %d iter.', ...
    R.m, R.v, size(R.TM,1)), 'Interpreter', 'latex');
legend('It. 1', 'It. last', 'It. 1: q^{\\i}', 'It. last: q^{\\i}');
% ylim([-30, 30])
grid on
hold off


end

