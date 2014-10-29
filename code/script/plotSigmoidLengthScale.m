function [ ] = plotSigmoidLengthScale(  )
%PLOTSIGMOIDLENGTHSCALE Investigate how much changing parameters of incoming messages
%affects the outgoing messages in sigmoid factor problem.
%   .

rng(10);
% x (Gaussian), z (Beta)
% messages from x 
n =  30;
mx = linspace(-14, 14, n);
vx = linspace(0.5, 50, n);
%vx = linspace(100, 300, n);

mmx = repmat(mx, [length(vx), 1]);
vvx = repmat(vx', [1, length(mmx)]);
%XX = DistNormal(mmx(:), vvx(:));
%X = DistNormal(mx, vs);
% messages from z 
z = DistBeta(2, 3);
%z = DistBeta(5, 10);
proposal = DistNormal(0, 15);
fac = @(x)(1./(1+exp(-x)));
N = length(mmx(:));
% number of importance weights to draw
K = 2e4;
tox_mean = zeros(size(mmx));
tox_var = zeros(size(vvx));
toz_mean = zeros(size(mmx));
toz_var = zeros(size(vvx));
tox_builder = DistNormalBuilder();
toz_builder = DistBetaBuilder();
reports = 1:floor(N/10):N;
for i=1:N
    x = DistNormal(mmx(i), vvx(i));
    xp = proposal.sampling0(K);
    zp = fac(xp);
    W = x.density(xp).*z.density(zp)./proposal.density(xp);
    WN = W/sum(W);

    xout = tox_builder.fromSamples(xp, WN);
    zout = toz_builder.fromSamples(zp, WN);

    tox_mean(i) = xout.mean;
    tox_var(i) = xout.variance;
    toz_mean(i) = zout.mean;
    toz_var(i) = zout.variance;
    if ismember(i, reports)
        display(sprintf('%.2g %% completed', 100*i/N));
    end

end


plotMeanOutputs(mmx, vvx, tox_mean, z);
title(sprintf('Report outgoing mean to x. Fix incoming m_z = Beta(%.2g, %.2g).', z.alpha, z.beta));
plotVarianceOutputs(mmx, vvx, tox_var, z);
title(sprintf('Report outgoing log variance to x. Fix incoming m_z = Beta(%.2g, %.2g).', z.alpha, z.beta));
plotMeanOutputs(mmx, vvx, toz_mean, z);
title(sprintf('Report outgoing mean to z. Fix incoming m_z = Beta(%.2g, %.2g).', z.alpha, z.beta));
plotVarianceOutputs(mmx, vvx, toz_var, z);
title(sprintf('Report outgoing log variance to z. Fix incoming m_z = Beta(%.2g, %.2g).', z.alpha, z.beta));
end

function plotMeanOutputs(mmx, vvx, to_mean, z)
    figure
    hold on
    %C=contourf(mmx, log(vvx), to_mean, 20);
    C=contourf(mmx./(vvx), -0.5./(vvx), to_mean, 20);
    clabel(C);
    colorbar
    set(gca, 'FontSize', 20);
    xlabel('Gaussian input: m/v');
    ylabel('Gaussian input -1/2v');
    %xlabel('Gaussian m/v');
    %ylabel('Gaussian log(precision)'); 
    grid on
    hold off 
end

function plotVarianceOutputs(mmx, vvx, to_var, z)
    figure
    hold on
    C=contourf(mmx./(vvx), -0.5./(vvx), log(to_var), 10);
    clabel(C);
    colorbar
    set(gca, 'FontSize', 20);
    xlabel('Gaussian input: m/v');
    ylabel('Gaussian input -1/2v');
    %xlabel('Gaussian input mean');
    %ylabel('Gaussian input log variance');
    grid on
    hold off 
end





