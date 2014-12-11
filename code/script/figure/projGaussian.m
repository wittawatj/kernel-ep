%function  projGaussian(  )
%PROJGAUSSIAN function to demonstrate a projection of an arbitrary distribution 
%on to the exponential family (Gaussian).
%

% arbitrary distribution
% mu (k x d)
mu = [2; 8];
sigma = [1];
mix = [0.25, 0.75];
p = gmdistribution(mu, sigma, mix);


gm_mean = mix*mu;
gm_var = var(random(p, [7e3]));

% plot
figure 
hold all 

% gmm 
X = linspace(-4, 15, 2e3);
lw = 4;
plot(X, pdf(p, X'), 'linewidth', lw );
plot(X, normpdf(X, gm_mean, gm_var^0.5), 'linewidth', lw);
set(gca, 'fontsize', 24);
%fplot(@(x)pdf(p, x), range, 'linewidth', 2);
%fplot(@(x)normpdf(x, gm_mean, gm_var^0.5), range);
%legend('Gaussian mixure', 'Projected to Gaussian');
legend('Tilted distribution r_{f \rightarrow V}', 'Projected to a Gaussian');
set(gca,'ytick',[]);
set(gca,'ycolor',[1 1 1])
%grid on
hold off 


%end

