function [ X, T, Xout, Tout ] = gentrain_dist2(X, T, op)
%GENTRAIN_DIST2 Generate training data for learning a conditional mean
%embedding operator mapping distributions to distribution.
% The function is for the case where the factor takes 2 incoming messages
% m_x and m_t and outputs m_out. Assume the factor is p(x|t).
%
% Xout, Tout contain outgoing messages without dividing by the cavity i.e.,
% q(.).
%

% Typically an array of DistNormal
assert(isa(X, 'Density'));
assert(isa(T, 'Density'));
assert(isa(X, 'Sampler'));
assert(isa(T, 'Sampler'));
assert(length(X) == length(T));
% xd = X(1).d;
% td = T(1).d;
N = length(X);

seed = myProcessOptions(op, 'seed', 1);

% importance sampling data size. K in Nicolas's paper.
iw_samples = myProcessOptions(op, 'iw_samples', 1e4);

% Importance weight vector can be a numerically zero vector when, for
% example, the messages have very small variance. iw_trials specifies the
% number of times to draw IW samples to try before giving up on the
% messages.
iw_trials = myProcessOptions(op, 'iw_trials', 20);

% Instead of samplilng from the in_proposal, if sample_cond_msg is true,
% then sample from mt (message from T) instead. T is the conditioned
% variable. If true, in_proposal is not needed and ignored.
sample_cond_msg = myProcessOptions(op, 'sample_cond_msg', false);

% The SSBuilder for x in p(x|t) i.e., the left variable. This SSBuilder will
% determine the output distribution of x.
% Output DistNormal by default.
left_ssbuilder = myProcessOptions(op, 'left_ssbuilder', DistNormal.getSSBuilder());
assert(isa(left_ssbuilder, 'DistBuilder'));

% The SSBuilder for t in p(x|t) i.e., the right variable. This SSBuilder will
% determine the output distribution of t.
% Output DistNormal by default.
right_ssbuilder = myProcessOptions(op, 'right_ssbuilder', DistNormal.getSSBuilder());
assert(isa(right_ssbuilder, 'DistBuilder'));

if ~sample_cond_msg
    % proposal distribution for for the conditional varibles (i.e. t)
    % in the factor. Require: Sampler & Density.
    in_proposal = op.in_proposal;
    assert(isa(in_proposal, 'Density'));
    assert(isa(in_proposal, 'Sampler'));
end

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
Xout = left_ssbuilder.empty(0, 1);
Tout = right_ssbuilder.empty(0, 1);
index = 1;
% Indices of bad messages
BadInd = [];
for i=1:N
    mx = X(i);
    mt = T(i);
    for j=1:iw_trials
        if sample_cond_msg
            % sample from mt instead of in_proposal
            TP = mt.sampling0(K);
            XP = cond_factor(TP);
            % compute importance weights
            W = mx.density(XP);
        else
            TP = in_proposal.sampling0(K);
            XP = cond_factor(TP);
            % compute importance weights
            W = mx.density(XP).*mt.density(TP) ./ in_proposal.density(TP);
        end
        
        assert( all(W >= 0));
        % projection. p(x|t)
        Xsuff = left_ssbuilder.suffStat(XP);
        Tsuff = right_ssbuilder.suffStat(TP);
        wsum = sum(W);
        WN = W/wsum;
%         WN = W/K;
%         error('/wsum or /K ?');
        xs = Xsuff*WN';
        ts = Tsuff*WN';
        
        if left_ssbuilder.stableSuffStat(xs) ...
                && right_ssbuilder.stableSuffStat(ts)
            
            % W be numerically 0 if the density values are too low.
            mx_out = left_ssbuilder.fromSuffStat(xs);
            mt_out = right_ssbuilder.fromSuffStat(ts);
            % store
            Xout(index) = mx_out;
            Tout(index) = mt_out;
            index = index + 1;
            break;
            
        else
            if j==iw_trials
                % not successful in getting nonzero W
                Xout(index) = left_ssbuilder.dummyObj();
                Tout(index) = right_ssbuilder.dummyObj();
                BadInd(end+1) = index;
                index = index+1;
            end

            % Assume mx and mt are somehow hard to deal with e.g., low variance.
            % Try again.
        end
        
    end
    
end

assert(length(X)>=length(Xout));
assert(length(T)>=length(Tout));

% exclude bad messages
X(BadInd) = [];
T(BadInd) = [];
Xout(BadInd) = [];
Tout(BadInd) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%
RandStream.setGlobalStream(oldRs);
end



