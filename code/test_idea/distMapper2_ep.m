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
display(sprintf('Starting EP in %s', mfilename));
% repeat until convergence
for t=1:ep_iters
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
        mxi_f = DistNormal(X(:,i), observed_variance);
        
        display(sprintf('x%d = %.2g', i, X(:,i)));
        q = mapper.mapDist2(mxi_f, qni);
        mfi_z = q/qni; %DistNormal division
        display(sprintf('m_f%d->theta = N(%.2g, %.2g)', i, mfi_z.mean, ...
            mfi_z.variance));
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

R.Mean = TM;
R.Variance = TV;
R.CavityMean = TMQNI;
R.CavityVariance = TVQNI;
R.q = q;


end

