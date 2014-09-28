%function  projGaussian(  )
%PROJGAUSSIAN function to demonstrate a projection of an arbitrary distribution 
%on to the exponential family (Gaussian).
%

% arbitrary distribution
% mu (k x d)
mu = [2; 8];
sigma = [1];
mix = [0.3, 0.7];
p = gmdistribution(mu, sigma, mix);


gm_mean = mix*mu;
gm_var = var(random(p, [1e4]));

% plot
figure 
hold all 

% gmm 
X = linspace(-3, 15, 2e3);
plot(X, pdf(p, X'), 'linewidth', 2 );
plot(X, normpdf(X, gm_mean, gm_var^0.5), 'linewidth', 2);
set(gca, 'fontsize', 20);
%fplot(@(x)pdf(p, x), range, 'linewidth', 2);
%fplot(@(x)normpdf(x, gm_mean, gm_var^0.5), range);
legend('Gaussian mixure', 'Projected to Gaussian');
grid on
hold off 


%end

