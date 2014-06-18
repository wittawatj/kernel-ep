function clutter_problem( seed )
% Clutter problem of Tom Minka
if nargin < 1
    seed = 1;
end
warning('%s may not work anymore due to the removal of many related files.')

oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed', seed);
RandStream.setGlobalStream(rs);
% parameters for clutter problem
a = 10;
w = 0.5;

N  = 1000;
% uniform prior
% fr = -10;
% to = 20;
% Tp = unifrnd(fr, to, 1, N);
% tp_pdf = @(t)unifpdf(t, fr, to);

% Gaussian prior
% mp = 1;
% vp = 40;
mp = 1;
vp = 30;
Tp = randn(1, N)*sqrt(vp) + mp;
tp_pdf = @(t)normpdf(t, mp, sqrt(vp));

[Xp] = ClutterMinka.x_cond_dist(Tp, a, w);
% Now we have a joint sample (Theta, X)

% new data set for testing EP. Not for learning an operator.
nN = 200;
[Theta, tdist] = Clutter.theta_dist(nN);
[NX, xdist] = ClutterMinka.x_cond_dist(Theta, a, w);

if false
    load('main/problem/clutter/clutter_data.mat');
    % Theta is also loaded.
    NX = observedData;
end

% ////// EP with Arthur's formula
% Learn mean embedding operator
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learn operator and EP iterations.
[Op]=learn_op(Tp, Xp);
[KR] = kernel_ep(NX, Op);
% [KR] = kernel_parallel_ep(NX, Op);
q = KR.q;

% ///// EP with my own formula
% Learn mean embedding operator
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learn operator and EP iterations.
% [Op_norw] = learn_op_no_reweights(Tp, Xp);
% [KR_norw] = kernel_ep_no_reweights(NX, Op_norw);
% q = KR_norw.q;


% Minka's EP
C = ClutterMinka(a, w);
% initial values for q
m0 = 0;
v0 = 10;
% TM (iterations x N) = mean of each i for each iteration t
[R] = C.ep(NX, m0, v0, seed );

%%% plot
plot_results(tdist, xdist, tp_pdf, q, seed);
% kernel EP
% [h1,h2]= plot_epfacs( KR_norw );
[h1,h2]= plot_epfacs( KR );
% [h1,h2]= plot_epfacs( R );

RandStream.setGlobalStream(oldRs);
keyboard
%%%%%%%%%%%%%%%%%%%%%%%%%%5
end


function [R] = kernel_ep_no_reweights(NX, Op)
% The factor in clutter problem has only two variables involved: X, theta.
% Here, we treat the factr as having three variables: X, theta, theta
% cavity. The idea is the treat the operator as taking two inputs X and
% theta from cavity, and mapping them to a message to theta.

nN = size(NX, 2);
f0 = DistNormal(0, 1000);
q = f0;
FT = DistNormal.empty(); %f tilde
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
for t=1:100
    pmean = q.mean;
    pvar = q.variance;
    display(sprintf('## EP iteration %d starts', t));
    
    for i=1:nN
        qni = q/FT(i); % DistNormal division
        % "skipping EP": If the cavity is not proper, skip index i
        if qni.variance < 0
            display(sprintf('Cavity q\\%d = N(%.2g, %.2g) not proper. Skip.', i, qni.mean, qni.variance));
            continue;
        end
        
        %         display(sprintf('Cavity q\\%d = N(%.2g, %.2g)', i, qni.mean, qni.variance));
        
        % we observed X. Use PointMass.
%                                 mxi_f = PointMass(NX(:,i));
        % we observed X. But, let's put a width around it
        mxi_f = DistNormal(NX(:,i), 0.1);
        
        %         display(sprintf('x%d = %.2g', i, NX(:,i)));
        
        qnew = Op.apply_pbp( mxi_f,  qni);
        mfi_z = qnew/qni;
        
        % variance cap. Ordinary EP also needs this.
        if mfi_z.variance > 1e4
            mfi_z = DistNormal(mfi_z.mean, 1e4);
        end
        %         display(sprintf('m_f%d->theta = N(%.2g, %.2g)', i, mfi_z.mean, ...
        %             mfi_z.variance));
        q = qnew;
        %         display(sprintf('q = N(%.2g, %.2g)', q.mean, q.variance));
        FT(i) = mfi_z;
        TMQNI(t, i) = qni.mean;
        TVQNI(t, i) = qni.variance;
        
        %         fprintf('\n');
    end
    % object array manipulation
    TM(t, :) = [FT.mean];
    TV(t, :) = [FT.variance];
    % check convergence
    if norm(q.mean-pmean)/norm(pmean) < 1e-3 ...
            && norm(q.variance - pvar,'fro')/norm(pvar, 'fro') < 1e-3
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
end % end method


function [R] = kernel_ep(NX, Op)

nN = size(NX, 2);
% prior factor for theta
f0 = DistNormal(0, 50);
% product of all incoming messages to theta. Estimate of posterior over
% theta
q = f0;
% f tilde's represented by DistNormal
FT = DistNormal.empty();
% initialize FT randomly based on f0
for i=1:nN
    FT(i)= DistNormal(0, Inf);
    %     FT(i) = DistNormal(f0.mean + randn(1)*10, 1e4);
    %     q = q*FT(i);
    
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
        
        qni = q/FT(i); % DistNormal division  
%         if qni.variance < 0
%             display(sprintf('Cavity q\\%d = N(%.2g, %.2g) not proper. Skip.', i, qni.mean, qni.variance));
%             continue;
%         end
        
        %         if qni.variance < 0.1
        %             positive but small => make it bigger
        %             qni = DistNormal(qni.mean, 0.1);
        %         end
        
        display(sprintf('Cavity q\\%d = N(%.2g, %.2g)', i, qni.mean, qni.variance));
        
        % we observed X. Use PointMass.
%                         mxi_f = PointMass(NX(:,i));
        % we observed X. But, let's put a width around it
        mxi_f = DistNormal(NX(:,i), 0.1);
        
        display(sprintf('x%d = %.2g', i, NX(:,i)));
        mfi_z = Op.apply_ep_approx( mxi_f,  qni);
%         mfi_z = Op.apply_bp( mxi_f);
        
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
R.m = q.mean;
R.v = q.variance;
end


function [R] = kernel_parallel_ep(NX, Op)

nN = size(NX, 2);
% prior factor for theta
f0 = DistNormal(0, 50);
% product of all incoming messages to theta. Estimate of posterior over
% theta
q = f0;
% f tilde's represented by DistNormal
FT = DistNormal.empty();
% initialize FT randomly based on f0
for i=1:nN
    FT(i)= DistNormal(0, Inf);
    %     FT(i) = DistNormal(f0.mean + randn(1)*10, 1e4);
    %     q = q*FT(i);
end

% TM = records of all means of f_i in each iteration
TM = [];
% TV = records of variance
TV = [];
TMQNI = [];
TVQNI = [];
% repeat until convergence
for t=1:4
    pmean = q.mean;
    pvar = q.variance;
    display(sprintf('## EP iteration %d starts', t));
    for i=1:nN
        
        qni = q/FT(i); % DistNormal division
        % Try "EP with skipping": If the cavity is not proper, skip index
        % i.
        if qni.variance < 0
            display(sprintf('Cavity q\\%d = N(%.2g, %.2g) not proper. Skip.', i, qni.mean, qni.variance));
            continue;
        end
        
        %         if qni.variance < 0.1
        %             positive but small => make it bigger
        %             qni = DistNormal(qni.mean, 0.1);
        %         end
        display(sprintf('Cavity q\\%d = N(%.2g, %.2g)', i, qni.mean, qni.variance));
        
        % we observed X. Use PointMass.
        mxi_f = PointMass(NX(:,i));
        % we observed X. But, let's put a width around it
        %         mxi_f = DistNormal(NX(:,i), 0.1);
        display(sprintf('x%d = %.2g', i, NX(:,i)));
        
        mfi_z = Op.apply_ep_approx( mxi_f,  qni);
        FT(i) = mfi_z;
        display(sprintf('m_f%d->theta = N(%.2g, %.2g)', i, mfi_z.mean, ...
            mfi_z.variance));
        
        
        TMQNI(t, i) = qni.mean;
        TVQNI(t, i) = qni.variance;
        
        fprintf('\n');
    end
    
    % object array manipulation
    TM(t, :) = [FT.mean];
    TV(t, :) = [FT.variance];
    
    q = f0;
    for i=1:nN
        q = q*FT(i);
        %     FT(i) = DistNormal(f0.mean + randn(1)*10, 1e4);
        %     q = q*FT(i);
    end
    % refresh posterior
    display(sprintf('q = N(%.2g, %.2g)', q.mean, q.variance));
    
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
R.m = q.mean;
R.v = q.variance;
end


function [Op]=learn_op(Tp, Xp)

% Now we have a joint sample (Theta, X)
% Learn mean embedding operator
%%%%%%%%%%%%%%%%%%%%%%%%%%

% operator at f_i from x to theta
% options = [];
% options.reglist  = [1e-4, 1e-2, 1];
% options.xwlist = [1/4, 1, 4];
% options.fold = 2;
% [Op] = CondOp1.learn_operator(Xp, Tp, options);

% options = [];
reglist = [1e-4, 1e-2, 1];
options.ksr_reglist = reglist;
options.kbr_reglist = reglist;
options.xwlist = [1/5, 1/2, 1, 2, 5];
options.fold = 2;
[Op, CVLog] = CondOp1.kbr_operator(Xp, Tp, options);

end


function [Op] = learn_op_no_reweights(Tp, Xp)
% for EP with no reweighting Beta vector

% Now we have a joint sample (Theta, X)
% Learn mean embedding operator
%%%%%%%%%%%%%%%%%%%%%%%%%%

% operator at f_i from x to theta
options = [];
options.reglist  = [1e-7, 1e-4, 1e-2, 1];
options.xwlist = [1/12, 1/8, 1/4, 1];
options.ywlist = [1/12, 1/8, 1/4, 1];
options.fold = 2;

% Explicitly treat theta samples from the cavity as input. So
% X = Xp;
% Y = Tp;
% Z = Tp;
[Op] = CondOp2.learn_operator(Xp, Tp, Tp, options);

end

