function [ ] = clutter_problem( seed )
% Clutter problem of Tom Minka
if nargin < 1
    seed = 1;
end

oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);          

N  = 400;
% from
fr = -7;
to = 8;
Tp = unifrnd(fr, to, 1, N);
tp_pdf = @(t)unifpdf(t, fr, to);

% Training dataset
%%%%%%%%%%%%%%%%%%%%%
[Xp] = x_cond_dist(Tp);

[d,N] = size(Xp);
% Now we have a joint sample (Theta, X)
% Learn mean embedding operator
%%%%%%%%%%%%%%%%%%%%%%%%%%

% operator at f_i from x to theta
options = [];
options.reglist  = [ 1e-2, 1, 10, 100];
options.xwlist = [1/2, 1, 2, 4, 8];
% options.xwlist = [1];
[Op] = CondOp1.learn_operator(Xp, Tp, options);

% EP iterations
%%%%%%%%%%%%%%%%%%%%%%
% new data set from the same distribution
nN = 400;
[Theta, tdist] = theta_dist(nN);
[NX, xdist] = x_cond_dist(Theta);

% prior factor for theta
f0 = DistNormal(2, 5);
% f tilde's represented by DistNormal
FT = cell(1, nN);
% product of all incoming messages to theta. Estimate of posterior over
% theta
q = f0;
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
        mxi_f = PointMass(NX(:,i));
        mfi_z = Op.apply_ep( mxi_f,  qni);
        
        q = qni*mfi_z;% DistNormal multiplication
        FT{i} = mfi_z;
    end
    % check convergence
    if norm(q.mean-qprev.mean)<1e-4 && norm(q.variance - qprev.variance, 'fro')<1e-4
%         break;
    end
    t
    q
end
% keyboard
RandStream.setGlobalStream(oldRs);


%%%%%%%%% plot
figure
hold on
set(gca, 'FontSize', 20)

range = [-8, 9];
dom = range(1):0.01:range(2);

% test theta distribution
plot(dom, tdist(dom), '-r', 'LineWidth', 2);
% distribution of x given mean theta 
fxdist = @(x)pdf(xdist, x');
plot(dom, fxdist(dom), '-b', 'LineWidth', 2);
% theta distribution used in training
plot(dom, tp_pdf(dom),  '-k', 'LineWidth', 2);

% q = resulting distribution of theta from EP
plot(dom, q.density(dom),  '-m', 'LineWidth', 2);


legend('Test \theta dist', 'X | mean \theta', 'Train \theta dist', 'q');
ylabel('Density')
xlim([range(1), range(2)])
ylim([0, .6])
title(sprintf('seed: %d. q: (mean, variance) = (%.3g, %.3g)', seed, q.mean, q.variance));
grid on

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%5
end

function [Theta, tdist] = theta_dist(N)
% Theta distribution on testing
var_theta = 1;
mu = 3;
dis_theta = DistNormal(mu, var_theta);
Theta = dis_theta.draw(N);
tdist = @(t)(normpdf(t, mu, var_theta ));
end

function [X, ftrue] = x_cond_dist(Theta)

a = 7;
% contamination rate w in [0,1]
w = 0.3;
N  = length(Theta);
cov(:,:,1) = 1;
cov(:,:,2) = a;

X = zeros(1, N);
F = cell(1,N);
for i=1:N
    theta = Theta(:,i);
    f = gmdistribution([theta; 0], cov, [1-w, w]);
    X(:,i) = f.random(1);
    F{i} = f;
end

ftrue =  gmdistribution([mean(Theta); 0], cov, [1-w, w]);
end

