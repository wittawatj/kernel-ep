classdef DistBuilder < handle & PrimitiveSerializable
    %DISTBuilder A Distribution object builder from samples or list of statistics.
    %    Subclasses should support Distribution and DistArray.
    
    properties
    end
    
    methods (Abstract)
        
        % Construct a statistic for each Distribution in D (array of
        % Distributions).
        % For example for a 1d Gaussian,
        % S = [x; x.^2]. S(:,i) is the statistic for D(i).
        % The statistic is meant 
        % to be some value which can be used as input in fromStat().
        % Basically, S represents D. 
        S=getStat(this, D);
        
        % construct a list of distributions D from a list of 
        % statistics (e.g., moments) S.
        % Typically S is what returns by getStat().
        D=fromStat(this, S)

        % Get a list of uncentred moments characterizing the Distribution D.
        % The output is a cell array and is such that this.fromMoments(Mcell) 
        % will give back D.
        % Mcell{i}{1} gives the first moment.
        % Mcell{i}{2} gives the second moment (a covariance matrix for multivariate
        % Distribution)  
        % And so on ...
        % where i is the Distribution index to support array of Distribution.
        %
        % Subclasses should ensure that Mcell can be used on other DistBuilder's
        % so that a set of moments from a Distribution can be used to construct
        % another Distribution of different parametric form.
        %
        Mcell=getMoments(this, D);

        % Construct a Distribution D from a cell array of moments Mcell.
        % Mcell is likely returned from this.getMoments(...).
        D=fromMoments(this, Mcell);
        
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

        s = shortSummary(this)
    end
    
end

