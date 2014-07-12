function [ X, T, Xout, Tout ] = gentrain_dist2(X, T, op)
%GENTRAIN_DIST2 Generate training data for learning a conditional mean
%embedding operator mapping distributions to distribution.
% The function is for the case where the factor takes 2 incoming messages
% m_x and m_t and outputs m_out. Assume the factor is p(x|t).
%
% Xout, Tout contain outgoing messages without dividing by the cavity i.e.,
% q(.).
%
% TODO: use multicore package to generate in parallel

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
iw_samples = myProcessOptions(op, 'iw_samples', 5e4);

% Importance weight vector can be a numerically zero vector when, for
% example, the messages have very small variance. iw_trials specifies the
% number of times to draw IW samples to try before giving up on the
% messages.
iw_trials = myProcessOptions(op, 'iw_trials', 5);

% Instead of samplilng from the in_proposal, if sample_cond_msg is true,
% then sample from mt (message from T) instead. T is the conditioned
% variable. If true, in_proposal is not needed and ignored.
sample_cond_msg = myProcessOptions(op, 'sample_cond_msg', false);

% The DistBuilder for x in p(x|t) i.e., the left variable. This SSBuilder will
% determine the output distribution of x. If [], Xout will be [].
% Output DistNormal by default.
if isfield(op, 'left_distbuilder')
    if isempty(op.left_distbuilder)
        left_distbuilder = [];
    else
        left_distbuilder = op.left_distbuilder;
    end
else
    left_distbuilder = DistNormal.getDistBuilder();
end
assert(isa(left_distbuilder, 'DistBuilder') || isempty(left_distbuilder));

% The DistBuilder for t in p(x|t) i.e., the right variable. This SSBuilder will
% determine the output distribution of t. If [], Tout will be [].
% Output DistNormal by default.
if isfield(op, 'right_distbuilder')
    if isempty(op.right_distbuilder)
        right_distbuilder = [];
    else
        right_distbuilder = op.right_distbuilder;
    end
else
    right_distbuilder = DistNormal.getDistBuilder();
end
assert(isa(right_distbuilder, 'DistBuilder') || isempty(right_distbuilder));

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
if ~isempty(left_distbuilder)
    Xout = left_distbuilder.empty(0, 1);
end

if ~isempty(right_distbuilder)
    Tout = right_distbuilder.empty(0, 1);
end
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
        wsum = sum(W);
        WN = W/wsum;
        % projection. p(x|t)
        if ~isempty(left_distbuilder)
            mx_out = left_distbuilder.fromSamples(XP, WN);
        end
        
        if ~isempty(right_distbuilder)
            mt_out = right_distbuilder.fromSamples(TP, WN);
        end
        
        if (isempty(left_distbuilder) || mx_out.isProper() )...
                && (isempty(right_distbuilder) || mt_out.isProper() )
            
            if ~isempty(left_distbuilder)
                Xout(index) = mx_out;
            end
            
            if ~isempty(right_distbuilder)
                Tout(index) = mt_out;
            end
            index = index + 1;
            break;
            
        else
            if j==iw_trials
                % not successful 
                BadInd(end+1) = index;
                % Assume mx and mt are somehow hard to deal with. Skip
            end
            % Try again
        end
        
    end
    
end

% exclude bad messages
X(BadInd) = [];
T(BadInd) = [];
if isempty(left_distbuilder)
   Xout=[];
end

if isempty(right_distbuilder)
    Tout = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
RandStream.setGlobalStream(oldRs);
end



