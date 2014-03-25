function [ ] = clutter_problem( )
% Clutter problem of Tom Minka
% variance of contamination distribution N(0, a*I)
N  = 400;
a = 1;
% contamination rate w in [0,1]
w = 0.3;
var_theta = 2;
% Training dataset
%%%%%%%%%%%%%%%%%%%%%
[Theta, X] = train_set(N, a, w, var_theta);

[d,N] = size(X);
% Now we have a joint sample (Theta, X)
% Learn mean embedding operator
%%%%%%%%%%%%%%%%%%%%%%%%%%

% operator at f_i from x to theta
options = [];
options.reglist  = [1e-4, 1e-2, 1, 10, 100];
options.xwlist = [1/2, 1, 2, 4, 8];
% options.xwlist = [1];
[Op] = CondOp1.learn_operator(X, Theta, options);


% EP iterations
%%%%%%%%%%%%%%%%%%%%%%
% new data set from the same distribution
ta = a;
tw = w;
tvar_theta = 0.5; 

[NTheta, NX] = train_set(400, ta, tw, tvar_theta);
[nd, nN] = size(NX);
% prior factor for theta
f0 = DistNormal(6, 5);
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
        break;
    end
    t
    q
end
% keyboard
%%%%%%%%%%%%%%%%%%%%%%%%%%5
end

function [Theta, X] = train_set(N, a, w, var_theta)


dis_theta = DistNormal(3, var_theta);
Theta = dis_theta.draw(N);
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

ftrue =  gmdistribution([dis_theta.mean; 0], cov, [1-w, w]);
fplot(@(x)(pdf(ftrue, x)), [-5, 8])
end