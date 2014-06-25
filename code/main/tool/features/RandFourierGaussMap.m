classdef RandFourierGaussMap < FeatureMap
    %RANDFOURIERGAUSSMAP Random Fourier map as in Rahimi & Recht for Gaussian kernel.
    %   - Input to genFeatures() is expected to be a vector/matrix.
    %   - dim must match the dimension of the input.

    properties (SetAccess=private)
        % Gaussian width squared
        gwidth2;
        % number of features to generate
        numFeatures;
        % dimension of the input
        dim;

        % weight matrix. dim x numFeatures
        W;
        % coefficients b. 1 x numFeatures. 
        % Drawn from U[0, 2*pi]
        B;
    end

    methods

        function this=RandFourierGaussMap(gwidth2, numFeatures, dim)
            this.gwidth2 = gwidth2; 
            assert(gwidth2 > 0);
            this.numFeatures = numFeatures;
            assert(numFeatures > 0);
            this.dim = dim;
            assert(dim > 0);
            this.W = randn(dim, numFeatures);
            this.B = randn(1, numFeatures)*2*pi;
        end

        function Z=genFeatures(this, X)  
            % X is a matrix of size dim x n  
            % Z = numFeatures x n

            assert(isnumeric(X));
            R = X/sqrt(this.gwidth2);
            d = this.dim;
            Z = cos(bsxfun(@plus, this.W'*R, this.B'))*sqrt(2/d);
        end

        function s=shortSummary(this)
            s = sprintf('RandFourierGaussMap(w^2=%.3f, #feat=%d)', ...
                this.gwidth2, this.numFeatures);
        end
    end

    methods (Access=private)
        
    end

end

