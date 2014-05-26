function clutter_egauss( seed )
% Clutter problem of Tom Minka solved by learning a conditional mean
% embedding operator for messages.
if nargin < 1
    seed = 1;
end

oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed', seed);
RandStream.setGlobalStream(rs);

% options
op.training_size = 800;
op.iw_samples = 1e4;
op.fold = 2;

% parameters for clutter problem
a = 10;
w = 0.5;
op.clutter_a = a;
op.clutter_w = w;

% CV
op.xwlist = [ 1/8, 1, 8, 16];
op.ywlist = [ 1/8, 1, 8, 16];
op.reglist = [1e-4, 1e-2, 1];

% new data set for testing EP. Not for learning an operator.
nN = 50;
[Theta, tdist] = Clutter.theta_dist(nN);
[NX, xdist] = ClutterMinka.x_cond_dist(Theta, a, w);
% if false
%     load('main/problem/clutter/clutter_data.mat');
%     % Theta is also loaded.
%     NX = observedData;
% end

% generate training set
[ X, T, Xout, Tout ] = gendata_clutter(op);
% Learn operator and EP iterations.. cross validation
Op = CondOpEGauss2.learn_operator(X, T, Tout, op);
% EP
[KR] = kernel_ep(NX, Op);
% [KR] = kernel_parallel_ep(NX, Op);
q = KR.q;


% % Minka's EP
% C = ClutterMinka(a, w);
% % initial values for q
% m0 = 0;
% v0 = 10;
% % TM (iterations x N) = mean of each i for each iteration t
% [R] = C.ep(NX, m0, v0, seed);
% 
% % Clutter problem solved with importance sampling projection
% theta_proposal = DistNormal(2, 20);
% iw_samples = 2e4;
% IC = ClutterImportance(a, w, theta_proposal, iw_samples);
% IR =  IC.ep( NX, m0, v0, seed);

%%% plot
% plot_clutter_results(tdist, xdist, tp_pdf, q, seed);
% kernel EP
% [h1,h2]= plot_epfacs( KR );
% [h1,h2]= plot_epfacs( R );

RandStream.setGlobalStream(oldRs);
keyboard
%%%%%%%%%%%%%%%%%%%%%%%%%%5
end



function [R] = kernel_ep(NX, Op)

nN = size(NX, 2);
% prior factor for theta
f0 = DistNormal(0, 30);
% product of all incoming messages to theta. Estimate of posterior over
% theta
q = f0;
% f tilde's represented by DistNormal
FT = DistNormal( zeros(1, nN), inf(1, nN));

% TM = records of all means of f_i in each iteration
TM = [];
% TV = records of variance
TV = [];
TMQNI = [];
TVQNI = [];
display(sprintf('Starting kernel_ep in %s', mfilename));
% repeat until convergence
for t=1:2
    pmean = q.mean;
    pvar = q.variance;
    display(sprintf('## EP iteration %d starts', t));
    for i=1:nN
        
        qni = q/FT(i); % DistNormal division  
%         if qni.variance < 0
%             display(sprintf('Cavity q\\%d = N(%.2g, %.2g) not proper. Skip.', i, qni.mean, qni.variance));
%             continue;
%         end
        
        % control the magnitude of the variance of the cavity
        if qni.variance < -1e1
            qni = DistNormal(qni.mean, -1e1);
        end
        
        if qni.variance > 100
            qni = DistNormal(qni.mean, 100);
        end
        
        display(sprintf('Cavity q\\%d = N(%.2g, %.2g)', i, qni.mean, qni.variance));
        
        % we observed X. Use PointMass.
%                         mxi_f = PointMass(NX(:,i));
        % we observed X. But, let's put a width around it
        mxi_f = DistNormal(NX(:,i), 0.1);
        
        display(sprintf('x%d = %.2g', i, NX(:,i)));
        mfi_z = Op.apply_ep( mxi_f,  qni);

        display(sprintf('m_f%d->theta = N(%.2g, %.2g)', i, mfi_z.mean, ...
            mfi_z.variance));
        
        q = qni*mfi_z;% DistNormal multiplication
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
    if ~q.isproper()
        break;
    end
end %end main for

R.TM = TM;
R.TV = TV;
R.TMQNI = TMQNI;
R.TVQNI = TVQNI;
R.q = q;
R.m = q.mean;
R.v = q.variance;
end



