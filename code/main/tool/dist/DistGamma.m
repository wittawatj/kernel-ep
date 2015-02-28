classdef DistGamma < handle & Distribution & HasKLDivergence
    %DISTGAMMA A Gamma distribution parameterized by shape and rate.
    
    properties (SetAccess = protected)
        mean;
        variance;
        
        % A cell array of parameters in the usual parametrization. For
        % Gaussian for example, this should be {mean, variance}.
        parameters;

        % dimension of the domain. For example, 1 for beta distribution.
        d;

        shape;
        rate;
    end
    
    methods
        function this=DistGamma(shape, rate)
            this.shape = shape;
            this.rate = rate;
            this.mean = shape./rate;
            this.variance = shape./rate.^2;
            this.d = 1;
            this.parameters = {shape, rate};
        end

        % Return true if this is a proper distribution e.g., positive
        % variance, not having inf mean, etc.
        function p=isProper(this)
            p = this.shape > 0 & this.rate > 0;
        end
        
        % Return the names in cell array corresponding to this.parameters
        % The cell array must have the same length as this.parameters.
        function names=getParamNames(this) 
            names = {'shape', 'rate'};
        end

        % Return the underlying Distribution type e.g., DistNormal
        function t=getDistType(this)
            t = mfilename;
        end

        function D=density(this, X)
            error('density of a gamma is not yet implemented.');
        end

        % Compute Hellinger distance from this distribution to the
        % distribution dq.
        function div=klDivergence(this, dq)
            assert(isa(dq, 'DistGamma'));
            assert(dq.isProper(), 'dq must be proper');
            ap = this.shape;
            bp = this.rate;
            aq = dq.shape;
            bq = dq.rate;
            div = (ap-aq)*psi(ap) - gammaln(ap) + gammaln(aq) + ...
                aq*(log(bp)-log(bq)) + ap*(bq-bp)/bp;

        end
            
    end
    
end

