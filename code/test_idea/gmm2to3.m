function [ ] = gmm2to3( )
% y ~ GMM(2 bumps )
% x|y ~ GMM(3 bumps with one bump depending on y)
% The goal is to infer the distribution of y given just the sample from x.

seed = 8;
oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);          

% data generation
Np=500; % number of points for training
N = 500;
% Gaussian prior for Y. Used for learning the operator
% m = 5;
% v = 20;
% Yp = randn(1, Np)*sqrt(v) + m;
% fp = @(y)pdf('Norm', y, m, v);

% Uniform prior for Y. Used for learning the operator
fr = 1;
to = 25;
Yp = unifrnd(fr, to, 1, Np);
fp = @(y)unifpdf(y, fr, to);

Xp = gen_data(Yp);
[Y,fy]=true_y_dist(N);
[X, ff]=gen_data(Y);

% operator at f_i from x to y
options = [];
options.reglist  = [1e-1, 1, 10, 100];
options.xwlist = [ 1/2, 1, 2, 4, 8];
options.fold = 5;
% options.xwlist = [1/8, 1/4, 1/2, 1, 2, 4, 6, 8];
% options.xwlist = [1];
% use Yp prior to learn
[Op] = CondOp1.learn_operator(Xp, Yp, options);


% EP iterations (BP ?)
%%%%%%%%%%%%%%%%%%%%%%
nN = size(X, 2);
% prior factor for Y
% f0 = DistNormal(m, v);
% f tilde's represented by DistNormal
FT = cell(1, nN);
% product of all incoming messages to y. Estimate of posterior over y
% q = f0;
q = DistNormal(0, 50); %flat prior
% repeat until convergence
for t=1:1
    qprev = q;
    for i=1:nN
        
        if isa(FT{i},'DistNormal')
            qni = q/FT{i}; % DistNormal division
        else
            qni = q;
        end
        % we observed X. Use PointMass.
        mxi_f = PointMass(X(:,i));
        mfi_z = Op.apply_ep( mxi_f,  qni);
        
        q = qni*mfi_z;% DistNormal multiplication
        FT{i} = mfi_z;
    end
    % check convergence
    if norm(q.mean-qprev.mean)<1e-4 && norm(q.variance - qprev.variance, 'fro')<1e-4
        break;
    end
    t
    q
end

% plot
figure
hold on
set(gca, 'FontSize', 20)

h1=ezplot(@(y)(pdf(fy,y)), [-2, 20]);
h2=ezplot(@(x)(pdf(ff, x)), [-5, 20]);
h3=ezplot(fp, [-2, 20]);
dom = -2:0.01:20;
h4=plot(dom, q.density(dom));

C = {h1, h2, h3,h4};
colors = {'r', 'b', 'k', 'm'};
for i=1:length(C)
    set(C{i}, 'LineWidth', 2, 'Color', colors{i});
end

legend('Y dist', 'X dist', 'Y prior', 'q');
ylabel('Density')
xlim([0 19])
ylim([0, .5])
title(sprintf('q: (mean, variance) = (%.3g, %.3g)', q.mean, q.variance));
hold off
grid on

% keyboard

%%%%%%%%%%%%%%%%%%%%%%%%%%5
RandStream.setGlobalStream(oldRs);


end

function [Y,f]=true_y_dist(N)

var = 0.4;
cov(:,:,1) = var;
cov(:,:,2) = var;
f = gmdistribution([4; 9], cov, [1/2, 1/2]);

Y = f.random(N)';
end

function [X, ff]=gen_data(Y)

ycov(:,:,1) = 0.5;
ycov(:,:,2) = 0.1;
% ycov(:,:,3) = 0.01;
w = 0.7;
% mixing = [w, 0.5-w/2, 0.5-w/2];
 mixing = [w, 1-w];
noise_mean = [14];
[d,N] = size(Y);
X = zeros(1,N);
for i=1:N
    yi = Y(:,i);
    f = gmdistribution([yi; noise_mean], ycov, mixing);
    X(:,i) = f.random(1);
end

ff =  gmdistribution([mean(Y); noise_mean], ycov, mixing);
% fplot(@(x)(pdf(ff, x)), [-5, 17])



end