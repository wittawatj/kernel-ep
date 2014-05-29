classdef DistBuilder < handle
    %DISTBuilder A Distribution object builder from samples.
    
    properties
    end
    
    methods (Abstract)
        
        % Construct a statistic for each Distribution in D (array of
        % Distributions).
        % For example for a 1d Gaussian,
        % S = [x; x.^2]. S(:,i) is the statistic for D(i).
        % In general, statistic returned may not be sufficient. It is meant
        % to be some values which can be taken as input in fromStat().
        % Basically, S represents D.
        S=getStat(this, D);
        
        % construct a list of distributions D from a list of 
        % statistics (e.g., moment parameters, expected sufficient statistic)
        % S. Typically S is what returned by getStat().
        D=fromStat(this, S)
        
        % Return a row vector T such that length(T)==size(S,2).
        % T(i)==true if S(:,i) is a stable and valid sufficient statistic
%         T=stableSuffStat(this, S)
        
        % Like calling className.empty(r, c). Useful for array
        % initialization.
        L=empty(this, r, c)
        
        % construct a Distribution object from the samples and
        % corresponding weights (on each instance). Weights can be from
        % importance weights, for instance.
        D= fromSamples(this, samples, weights)
    end
    
end

