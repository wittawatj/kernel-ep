classdef DistBuilder
    %DISTBuilder A Distribution object builder from samples.
    
    properties
    end
    
    methods (Abstract)
        
        % Construct a list of sufficient statistics from the data matrix X.
        % Each column in X is one instance. For example for a 1d Gaussian,
        % S = [x; x.^2]. S(:,i) is the sufficient statistic for X(:, i).
%         S=suffStat(this, X);
        
        % construct a list of distributions D from a list of sufficient
        % statistics (or moment parameters for expected sufficient statistic)
        % S. S is what returned by suffStat.
%         D=fromSuffStat(this, S)
        
        % Return a row vector T such that length(T)==size(S,2).
        % T(i)==true if S(:,i) is a stable and valid sufficient statistic
%         T=stableSuffStat(this, S)
        
        % Like calling className.empty(r, c). Useful for array
        % initialization.
        L=empty(this, r, c)
        
        % Construct one dummy object. 
%         obj=dummyObj(this)
        
        % construct a Distribution object from the samples and
        % corresponding weights (on each instance). Weights can be from
        % importance weights, for instance.
        D= fromSamples(this, samples, weights)
    end
    
end

