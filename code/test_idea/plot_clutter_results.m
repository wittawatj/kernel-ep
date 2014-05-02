function plot_clutter_results(test_theta, xdist, train_theta, q, seed)

%%%%%%%%% plot
figure
hold on
set(gca, 'FontSize', 20)

range = [-8, 9];
dom = range(1):0.01:range(2);

% test theta distribution
plot(dom, test_theta(dom), '-r', 'LineWidth', 2);
% distribution of x given mean theta
fxdist = @(x)pdf(xdist, x');
plot(dom, fxdist(dom), '-b', 'LineWidth', 2);
% theta distribution used in training
plot(dom, train_theta(dom),  '-k', 'LineWidth', 2);

% q = resulting distribution of theta from EP
plot(dom, q.density(dom),  '-m', 'LineWidth', 2);


legend('Test \theta dist', 'X | mean \theta', 'Train \theta dist', 'q');
% legend('Test \theta dist', 'X | mean \theta');
ylabel('Density')
xlim([range(1), range(2)])
ylim([0, .6])
title(sprintf('seed: %d. q: (mean, variance) = (%.3g, %.3g)', seed, q.mean, q.variance));
grid on

hold off
end
