classdef RFGEProdMap < FeatureMap
    %RFGEPRODMAP Random Fourier features for expected product kernel 
    %using Gaussian kernel for mean embeddings.
    %    - Input is a Distribution or a DistArray
    %    - If the input Distribution is not a DistNormal, then convert it to one 
    %    by moment matching. Computing a random features for an ExpFam Distribution
    %    amounts to computing expectation of cos() under the distribution. 
    %    To avoid deriving E[cos()] for every new Distribution type, we simply 
    %    convert it to a DistNormal.
    %
    
    properties(SetAccess=protected)
        % Gaussian width^2 for the embedding kernel
        gwidth2;
        % number of random features
        numFeatures;

        % weight matrix. dim x numFeatures
        W;
        % coefficients b. 1 x numFeatures. 
        % Drawn from U[0, 2*pi]
        B;
    end
    
    methods

        function this=RFGEProdMap(gwidth2, numFeatures)
            assert(gwidth2>0);
            assert(numFeatures>0);
            this.gwidth2=gwidth2;
            this.numFeatures=numFeatures;
            % W,B will be initialized the first time features are generated.
            % dimension of W is determined at that time.
            this.W=[];
            this.B=[];
        end

        function Z=genFeatures(this, D)
            % Z = numFeatures x n
            % Support multivariate DistNormal
            assert(isa(D, 'Distribution') || isa(D, 'DistArray'));
            this.initMap(D);
            n=length(D);
            %M=[D.mean];
            [M, V]=RFGEProdMap.getMVs(D);
            % always make V 3d
            if size(V, 1)==1
                V=shiftdim(V, -1);
                assert(ndims(V)==3);
            end

            Z=zeros(this.numFeatures, n );
            W=this.W;
            B=this.B;
            C=cos(bsxfun(@plus, W'*M, B')); % numFeatures x n
            for j=1:n
                S=sum( (W'*V(:,:,j)).*W', 2);
                E=exp(-0.5*S);
                Z(:, j)=E;
            end
            Z=Z.*C*sqrt(2/this.numFeatures);
        end

        function M=genFeaturesDynamic(this, D)
            % Generate feature vectors in the form of DynamicMatrix.
            assert(isa(D, 'Distribution') || isa(D, 'DistArray'));
            g=this.getGenerator(D);
            n=length(D);
            M=DefaultDynamicMatrix(g, this.numFeatures, n);
        end

        function g=getGenerator(this, D)
            g=@(I, J)this.generator(D, I, J);

        end

        function Z=generator(this, D, I, J )
            % I=indices of features, J=sample indices 
            assert(isa(D, 'Distribution') || isa(D, 'DistArray'));

            this.initMap(D);
            % array of Distribution's
            DJ=D(J);
            [M, V]=RFGEProdMap.getMVs(DJ);
            % always make V 3d
            if size(V, 1)==1
                V=shiftdim(V, -1);
                assert(ndims(V)==3);
            end

            Z=zeros(length(I), length(J) );
            W=this.W(:, I);
            B=this.B(:, I);
            C=cos(bsxfun(@plus, W'*M, B')); % length(I) x length(J)
            for j=1:length(J)
                S=sum( (W'*V(:,:,j)).*W', 2);
                E=exp(-0.5*S);
                Z(:, j)=E;
            end
            Z=Z.*C*sqrt(2/this.numFeatures);
        end

        function s=shortSummary(this)
            s = sprintf('%s(w^2=%.3f, #feat=%d)', ...
                mfilename, this.gwidth2, this.numFeatures);
        end

        %function s=saveobj(this)

        %end
    end

    methods(Access=private)
        function initMap(this, D)
            % Initialize the map only once. 
            if isempty(this.W) 
                assert(isempty(this.B));
                % Fourier transform of a Gaussian kernel is a Gaussian with 
                % reciprocal width.
                assert(length(unique([D.d]))==1);
                dim=unique([D.d]); % .d from Distribution interface 
                this.W=randn(dim, this.numFeatures)/sqrt(this.gwidth2);
                this.B=rand(1, this.numFeatures)*2*pi;

            end
        end
    end
    
    methods (Static)
        %function obj=loadobj(s)
        %end

        function map=createFromWeights(W, B)
            % Create an RFGEProdMap by manually specifying weight matrix W and B.
            % W is dim x numFeatures
            % B is 1 x numFeatures
            assert(isnumeric(W));
            assert(isnumeric(B));
            assert(size(W, 2)==size(B, 2));
            nf=size(W, 2);

            gwidth2=inf();
            map=RFGEProdMap(gwidth2, nf);
            map.W=W;
            map.B=B;
        end

        function [M, V]=getMVs(D)
            % Get means and variances of the Distribution's
            assert(isa(D, 'Distribution') || isa(D, 'DistArray'));
            dims=[D.d];
            isMulti=dims(1)>1;
            M=[D.mean];
            if isMulti
                V=cat(3, D.variance);
                assert(size(V, 1)>1);
            else
                V=[D.variance];
            end
        end

        function N=toDistNormal(D)
            % Convert Distribution (not DistArray) to DistNormal by moment matching 
            % D can be an array of Distribution's
            assert(isa(D, 'Distribution') );
            if isa(D, 'DistNormal')
                N=D;
                return;
            end
            fromBuilder=D.getDistBuilder();
            Mcell=fromBuilder.getMoments(D);
            toBuilder=DistNormalBuilder();
            N=toBuilder.fromMoments(Mcell);
            assert(isa(N, 'DistNormal'));
            assert(length(N)==length(D));

        end

    end
end

