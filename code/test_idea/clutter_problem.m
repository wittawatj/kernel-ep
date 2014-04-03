function [ ] = clutter_problem( seed )
% Clutter problem of Tom Minka
if nargin < 1
    seed = 1;
end

oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);
% parameters for clutter problem
a = 10;
w = 0.5;

N  = 1500;
% from
% fr = -7;
% to = 8;
fr = -2;
to = 6;
Tp = unifrnd(fr, to, 1, N);
tp_pdf = @(t)unifpdf(t, fr, to);

% Now we have a joint sample (Theta, X)
% Learn mean embedding operator
%%%%%%%%%%%%%%%%%%%%%%%%%%
[Op]=learn_op(Tp, a, w);

% EP iterations
%%%%%%%%%%%%%%%%%%%%%%
% new data set from the same distribution
nN = 20;
[Theta, tdist] = theta_dist(nN);
[NX, xdist] = ClutterMinka.x_cond_dist(Theta, a, w);

[KR] = kernel_ep(NX, Op);

% Minka's EP
C = ClutterMinka(a, w);
% initial values for q
m0 = 0;
v0 = 10;
% TM (iterations x N) = mean of each i for each iteration t
[R] = C.ep(Theta, m0, v0, seed );

q = KR.q;
% plot
plot_results(tdist, xdist, tp_pdf, q, seed);


RandStream.setGlobalStream(oldRs);

keyboard
%%%%%%%%%%%%%%%%%%%%%%%%%%5
end

function [R] = kernel_ep(NX, Op)

nN = size(NX, 2);
% prior factor for theta
f0 = DistNormal(0, 10);
% product of all incoming messages to theta. Estimate of posterior over
% theta
q = f0;
% f tilde's represented by DistNormal
FT = DistNormal.empty();
% initialize FT randomly based on f0
for i=1:nN
    FT(i)= DistNormal(0, Inf);
end

% TM = records of all means of f_i in each iteration
TM = [];
% TV = records of variance 
TV = [];
TMQNI = [];
TVQNI = [];
% repeat until convergence
for t=1:2
    pmean = q.mean;
    pvar = q.variance;
    display(sprintf('## EP iteration %d starts', t));
    for i=1:nN
        
        if isa(FT(i),'DistNormal')
            qni = q/FT(i); % DistNormal division
        else
            qni = q;
        end
        
        % qni cavity can be ill-formed if for example, q and FT{i} have
        % equal precision, then the resulting qni will have 0 precision or
        % Inf variance.
        % Try "EP with skipping": If the cavity is not proper, skip index
        % i.
        
        if qni.variance < 0
            display(sprintf('Cavity q\\%d = N(%.2g, %.2g) not proper. Skip.', i, qni.mean, qni.variance));
            continue;
        elseif qni.variance < 0.1
            % positive but small => make it bigger
            qni = DistNormal(qni.mean, 0.1);
        end
        
%         if isnan(qni.variance) || qni.variance < 1e-4
% %             display(sprintf('Cavity q\\%d = N(%.2g, %.2g) not proper. Skip.', i, qni.mean, qni.variance));
% %             continue;
%             qni = DistNormal(qni.mean, 1e-4);
%         end
        
        display(sprintf('Cavity q\\%d = N(%.2g, %.2g)', i, qni.mean, qni.variance));
        
        % we observed X. Use PointMass.
        mxi_f = PointMass(NX(:,i));
        display(sprintf('x%d = %.2g', i, NX(:,i)));
        mfi_z = Op.apply_ep_approx( mxi_f,  qni);
        display(sprintf('m_f%d->theta = N(%.2g, %.2g)', i, mfi_z.mean, ...
            mfi_z.variance));
        
        q = qni*mfi_z;% DistNormal multiplication
        % set a lower bound on the variance of q (upper bound on precision)
        % for numerical stability
        %         if q.variance < 1e-2
        %             q = DistNormal(q.mean, 1e-2);
        %         end
        display(sprintf('q = N(%.2g, %.2g)', q.mean, q.variance));
        FT(i) = mfi_z;
        TMQNI(t, i) = qni.mean;
        TVQNI(t, i) = qni.variance;
        
        fprintf('\n');
    end
    % object array manipulation
    TM(t, :) = [FT.mean];
    TV(t, :) = [FT.variance];
    % check convergence
    if norm(q.mean-pmean)<1e-2 && norm(q.variance - pvar, 'fro')<1e-2
                break;
    end
    
    
end %end main for

R.TM = TM;
R.TV = TV;
R.TMQNI = TMQNI;
R.TVQNI = TVQNI;
R.q = q;
end

function [Op]=learn_op(Tp, a, w)

% Training dataset
%%%%%%%%%%%%%%%%%%%%%
[Xp] = ClutterMinka.x_cond_dist(Tp, a, w);

% Now we have a joint sample (Theta, X)
% Learn mean embedding operator
%%%%%%%%%%%%%%%%%%%%%%%%%%

% operator at f_i from x to theta
options = [];
options.reglist  = [ 1e-4,  1e-2, 1, 10];
options.xwlist = [1/10, 1/5, 1, 2, 4];
options.fold = 2;
% options.xwlist = [1];
[Op] = CondOp1.learn_operator(Xp, Tp, options);

end


function plot_results(test_theta, xdist, train_theta, q, seed)

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
ylabel('Density')
xlim([range(1), range(2)])
ylim([0, .6])
title(sprintf('seed: %d. q: (mean, variance) = (%.3g, %.3g)', seed, q.mean, q.variance));
grid on

hold off
end

function [Theta, tdist] = theta_dist(N)
% Theta distribution on testing
var_theta = 1;
mu = 3;
dis_theta = DistNormal(mu, var_theta);
Theta = dis_theta.draw(N);
tdist = @(t)(normpdf(t, mu, var_theta ));
end



