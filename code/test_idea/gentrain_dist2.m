function [ X, T, Xout, Tout ] = gentrain_dist2(X, T, op)
%GENTRAIN_DIST2 Generate training data for learning a conditional mean
%embedding operator mapping distributions to distribution.
% The function is for the case where the factor takes 2 incoming messages
% m_x and m_t and outputs m_out. Assume the factor is p(x|t).
%
% Xout, Tout contain outgoing messages without dividing by the cavity i.e.,
% q(.).

% Typically an array of DistNormal
assert(isa(X, 'Density'));
assert(isa(T, 'Density'));
assert(length(X) == length(T));
xd = X(1).d;
td = T(1).d;
N = length(X);

seed = myProcessOptions(op, 'seed', 1);

% importance sampling data size. K in Nicolas's paper.
iw_samples = myProcessOptions(op, 'iw_samples', 1e4);

% Importance weight vector can be a numerically zero vector when, for
% example, the messages have very small variance. iw_trials specifies the
% number of times to draw IW samples to try before giving up on the
% messages.
iw_trials = myProcessOptions(op, 'iw_trials', 20);
% proposal distribution for for the conditional varibles (i.e. t)
% in the factor. Require: Sampler & Density.
in_proposal = op.in_proposal;
assert(isa(in_proposal, 'Density'));
assert(isa(in_proposal, 'Sampler'));

% A forward sampling function taking samples (array) from in_proposal and
% outputting samples from the conditional distribution represented by the
% factor.
cond_factor = op.cond_factor;

% change seed
oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed', seed);
RandStream.setGlobalStream(rs);
% number of importance-weighted samples.
K = iw_samples;

% outputs
Xout = DistNormal.empty(0, 1);
Tout = DistNormal.empty(0, 1);
index = 1;

for i=1:N
    mx = X(i);
    mt = T(i);
    for j=1:iw_trials
        
        TP = in_proposal.sampling0(K);
        
        XP = cond_factor(TP);
        % compute importance weights
        W = mx.density(XP).*mt.density(TP) ./ in_proposal.density(TP);
        
        assert( all(W >= 0));
        % projection
        Xsuff = DistNormal.normalSuffStat(XP);
        Tsuff = DistNormal.normalSuffStat(TP);
        wsum = sum(W);
        WN = W/wsum;
%         WN = W/K;
%         error('/wsum or /K ?');
        xs = Xsuff*WN';
        ts = Tsuff*WN';
        
        xmean = xs(1:xd);
        xvar = reshape(xs((xd+1):end), xd, xd) - xmean*xmean';
        tmean = ts(1:td);
        tvar = reshape(ts((td+1):end), td, td) - tmean*tmean';
        
        if all(~isnan(xmean)) && all(~isnan(xvar)) && ...
                all(~isnan(tmean)) && all(~isnan(tvar)) && ...
                all(abs(xvar) > 1e-4) && all(abs(tvar) > 1e-4)
            % W be numerically 0 if the density values are too low.
            mx_out = DistNormal(xmean, xvar);
            mt_out = DistNormal(tmean, tvar);
            % store
            Xout(index) = mx_out;
            Tout(index) = mt_out;
            index = index + 1;
            break;
            
        else
            if j==iw_trials
                % not successful in getting nonzero W
                Xout(index) = DistNormal(nan(xd, 1), inf(xd, 1));
                Tout(index) = DistNormal(nan(xd, 1), inf(xd, 1));
                index = index+1;
            end
            % Assume mx and mt are somehow hard to deal with e.g., low variance.
            % Try again.
        end
        
    end
    
end

assert(length(X)==length(Xout));
assert(length(T)==length(Tout));

% exclude bad messages
I = any( isnan([Xout.mean]), 1) ;
X(I) = [];
T(I) = [];
Xout(I) = [];
Tout(I) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%
RandStream.setGlobalStream(oldRs);
end



