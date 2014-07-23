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
            Z = cos(bsxfun(@plus, this.W'*R, this.B'))*sqrt(2/this.numFeatures);
        end

        function g=getGenerator(this, X)
            g=@(I, J)this.generator(X, I, J);

        end

        function M=generator(this, X, I, J )
            assert(isnumeric(X));
            R = X/sqrt(this.gwidth2);
            WT = this.W';
            subW = WT(I, :); %numFeatures x dim
            subB = this.B(I)';
            M = cos(bsxfun(@plus, subW*R(:, J), subB))*sqrt(2/this.numFeatures);
        end

        function M=genFeaturesDynamic(this, X)
            assert(isa(X, 'DistArray') || isa(X, 'TensorInstances'));
            g=this.getGenerator(X);
            n=X.count();
            M=DefaultDynamicMatrix(g, this.numFeatures, n);
        end

        function s=shortSummary(this)
            s = sprintf('%s(w^2=%.3f, #feat=%d)', ...
                mfilename, this.gwidth2, this.numFeatures);
        end

        function s=saveobj(this)
            s.gwidth2=this.gwidth2;
            s.numFeatures=this.numFeatures;
            s.dim=this.dim;
            s.W=this.W;
            s.B=this.B;
        end
    end

    methods(Static)
        function obj=loadobj(s)
            % constructor will draw W, B. Inefficient because they will be 
            % replaced anyway.
            obj=RandFourierGaussMap(s.gwidth2, s.numFeatures, s.dim);
            % Whatever... for now
            obj.W=s.W;
            obj.B=s.B;
        end
    end


end

