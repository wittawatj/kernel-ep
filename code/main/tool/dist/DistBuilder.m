classdef DistBuilder
    %DISTBuilder A distribution builder from sufficient statistic.
    %   An object of this class is intended to be created by call an
    %   appropriate static method of the distribution supporting
    %   DistBuilder.
    
    properties
    end
    
    methods (Abstract)
        
        % Construct a list of sufficient statistics from the data matrix X.
        % Each column in X is one instance. For example for a 1d Gaussian,
        % S = [x; x.^2]. S(:,i) is the sufficient statistic for X(:, i).
        S=suffStat(this, X);
        
        % construct a list of distributions D from a list of sufficient
        % statistics S. S is what returned by suffStat.
        D=fromSuffStat(this, S)
        
        % Return a row vector T such that length(T)==size(S,2).
        % T(i)==true if S(:,i) is a stable and valid sufficient statistic
        T=stableSuffStat(this, S)
        
        % Like calling className.empty(r, c)
        L=empty(this, r, c)
        
        % Construct one dummy object. 
        obj=dummyObj(this)
    end
    
end

