classdef Grid1DSampler < Density & Sampler
    %GRID1DSAMPLER Draws real numbers which are uniformly spaced 
    
    properties
        from;
        to;
    end
    
    methods
        function this=Grid1DSampler(f, t)
            assert(isnumeric(f));
            assert(length(f) == 1);
            assert(isnumeric(t));
            assert(length(t) == 1);
            this.from = f;
            this.to = t;
        end

        function D=density(this, X)
            D = unifpdf(X, this.from, this.to);
        end

        % Return samples X=[x_1, x_2, ...] 
        function X = sampling0(this, N)
            X = linspace(this.from, this.to, N);
        end
    end
    
end

