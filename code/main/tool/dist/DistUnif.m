classdef DistUnif < Distribution & Sampler & Density
    %DISTUNIF A uniform distribution

    properties (SetAccess=protected)
        mean;
        variance;
        
        % A cell array of parameters in the usual parametrization. For
        % Gaussian for example, this should be {mean, variance}.
        parameters;

        % dimension of the domain. 1 for uniform distributions.
        d;

        from;
        to;
    end
    
    methods 
        function this=DistUnif(f, t)
            assert(isnumeric(f));
            assert(isnumeric(t));
            assert(all(f < t));
            this.mean = (f+t)/2.0;
            this.variance = (t-f).^2/12.0;
            this.from = f;
            this.to = t;
            this.d = 1;
            this.parameters = {f, t};
        end

        % Return true if this is a proper distribution e.g., positive
        % variance, not having inf mean, etc.
        function p=isProper(this)
            p = isfinite(this.from) && isfinite(this.to);
        end

        % Return the names in cell array corresponding to this.parameters
        % The cell array must have the same length as this.parameters.
        function names=getParamNames(this)
            names={'from', 'to'};
        end
            

        % Return the underlying Distribution type e.g., DistNormal
        function t=getDistType(this)
            t = mfilename;
        end

        function D=density(this, X)
            D = unifpdf(X, this.from, this.to);
        end
        % Return samples X=[x_1, x_2, ...] 
        function X = sampling0(this, N)
            X = unifrnd(this.from, this.to, 1, N);
        end
    end
    
    methods (Static)
        function builder = getDistBuilder()
            builder=[];
            error('Subclass of Distribution needs to override this.');
            
        end
    end
end

