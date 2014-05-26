function [ R] = distMapper2_ep( X, mapper, op)
%DISTMAPPER2_EP A generic function to test EP with a DistMapper2
%    - Assume that the DistMapper2 takes two inputs X and T and that it
%    maps to T'.
%    - In the context of EP, the second argument T would be the cavity.
%    - After the mapping, the output distribution is divided by T (cavity)
%    to get a message from the factor to T.
%    - Assume all incoming and outgoing messages are DistNorml.
% Input:
%    - X is matrix of observed values where each column is one value.
% Return:
%    - A structure logging mean and variance of each factor during EP
%
assert(isnumeric(X));
assert(isa(mapper, 'DistMapper2'));
[d, nN] = size(X);
assert(d==1, 'Only for univariate Gaussian for now');

% prior distribution on T i.e., the variable to be inferred.
f0 = myProcessOptions(op, 'f0', DistNormal(0, 100));

% maximum number of EP sweeps to do
ep_iters = myProcessOptions(op, 'ep_iters', 10);

% Instead of using a PointMass for observed values, we use DistNormal. This
% parameter specifies the variance of the distribution. Typically small.
observed_variance = myProcessOptions(op, 'observed_variance', 0.1);

% convergence threshold for the mean of the posterior
mean_conv_thresh = myProcessOptions(op, 'mean_conv_thresh', 1e-2);
assert(mean_conv_thresh >= 0);

% convergence threshold for the variance of the posterior
var_conv_thresh = myProcessOptions(op, 'var_conv_thresh', 1e-2);
assert(var_conv_thresh >= 0);

% ######### Begin EP #####

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
% records of q in each sweep
Q = DistNormal.empty(0, 1);
display(sprintf('Starting EP in %s', mfilename));
% repeat until convergence
for t=1:ep_iters
    pmean = q.mean;
    pvar = q.variance;
    display(sprintf('## EP iteration %d starts', t));
    for i=1:nN
        skip = false;
        qni = q/FT(i); % DistNormal division
        
        
        % skip if the cavity is ill-scaled
        if abs(qni.variance) > 1e3
            display(sprintf('Cavity q\\%d = N(%.2g, %.2g) ill-scaled. Skip.', i, qni.mean, qni.variance));
            skip = true;
        end
        % we observed X. Use PointMass.
        %                         mxi_f = PointMass(NX(:,i));
        % we observed X. But, let's put a width around it
        if ~skip
            display(sprintf('Cavity q\\%d = N(%.2g, %.2g)', i, qni.mean, qni.variance));
            mxi_f = DistNormal(X(:,i), observed_variance);
            
            display(sprintf('x%d = %.2g', i, X(:,i)));
            qnew = mapper.mapDist2(mxi_f, qni);
            if abs(qnew.variance) > 1e4 || abs(qnew.variance) < 1e-2
                display(sprintf('Ill-scaled qnew = N(%.2g, %.2g)', qnew.mean, qnew.variance));
                skip = true;
            end
            
            if ~skip
                mfi_z = qnew/qni; %DistNormal division
                mv = [mfi_z.mean, mfi_z.variance];
                if any(isinf(mv)) || any(isnan(mv))
                    display(sprintf('f_%d  = N(%.2g, %.2g) not proper. Skip.', i, mfi_z.mean, mfi_z.variance));
                    skip = true;
                end
            end
        end
        
        if ~skip
            % control the magnitude of the variance of the mfi_z
            if abs(mfi_z.variance) > 1e4
                mfi_z = DistNormal(mfi_z.mean, sign(mfi_z.variance)*1e4 );
            elseif abs(mfi_z.variance) < 1e-2
                mfi_z = DistNormal(mfi_z.mean, sign(mfi_z.variance)*1e-2 );
            end
            q = qnew;
            display(sprintf('m_f%d-> = N(%.2g, %.2g)', i, mfi_z.mean, ...
                mfi_z.variance));
            display(sprintf('q = N(%.2g, %.2g)', q.mean, q.variance));
            
            FT(i) = mfi_z;
            TMQNI(t, i) = qni.mean;
            TVQNI(t, i) = qni.variance;
            
        else
            % skip
            TMQNI(t, i) = nan;
            TVQNI(t, i) = nan;
        end
        fprintf('\n');
    end
    % object array manipulation
    TM(t, :) = [FT.mean];
    TV(t, :) = [FT.variance];
    Q(t) = DistNormal(q.mean, q.variance);
    % check convergence
    if norm(q.mean-pmean)<mean_conv_thresh && ...
            norm(q.variance - pvar, 'fro')<var_conv_thresh
        break;
    end
    
end %end main for

R.Mean = TM;
R.Variance = TV;
R.CavityMean = TMQNI;
R.CavityVariance = TVQNI;
R.q = q;
% R.Factors = FT;
R.Q = Q;


end

