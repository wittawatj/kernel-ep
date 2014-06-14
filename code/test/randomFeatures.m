function randomFeatures()
% Test random features by Rahimi and Recht

% original dimension
d = 10;

% testing points
n = 1e3;
X = rand(d, n)  + poissrnd(0.3, d, n);
gwidth2 = meddistance(X)^2
% gwidth2 = 3;
X = X/sqrt(gwidth2);

% try multiple D.
Ds = 500:500:5000;
MaxErr = zeros(1, length(Ds)); 
AvgErr = zeros(1, length(Ds));
SDErr = zeros(1, length(Ds));
for i=1:length(Ds)
    % dimension of random features
    D = Ds(i);
    
    W = randn(d, D);
    B = rand(1, D)*2*pi;
    Z = zfunc(X, W, B, gwidth2); % Dxn
    Kran = Z'*Z;
    
    % real kernel matrix. Gaussian kernel with width = 1
    K = kerGaussian(X, X, 1);
    
    Diff = abs(Kran-K);
    AvgErr(i) = mean(Diff(:));
    MaxErr(i) = max(Diff(:));
    SDErr(i) = std(Diff(:));
end

% plot
hold on
plot(Ds, AvgErr, '-ro', 'linewidth', 2);
plot(Ds, MaxErr, '-bo', 'linewidth', 2);
plot(Ds, SDErr, '-ko', 'linewidth', 2);
set(gca, 'fontsize', 20);
xlabel('#Random features');
% ylabel('Norm of difference');
ylabel('difference');
title('Difference of approximated kernel matrix and the true.');
legend('Avg diff.', 'max entry diff', 's.d. entry diff');
grid on;
hold off

% imagesc(abs(Kran-K));
% colorbar
% keyboard;

end

function Z=zfunc(X, W, B, gwidth2)
D = size(W, 2);
Z = cos(bsxfun(@plus, W'*X, B'))*sqrt(2/D);
end


