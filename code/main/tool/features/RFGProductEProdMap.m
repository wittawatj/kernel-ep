classdef RFGProductEProdMap < FeatureMap & PrimitiveSerializable
    %RFGPRODUCTEPRODMAP Random Fourier features for product of expected product 
    %kernel using Gaussian kernel for mean embeddings. 
    %    - Input is a TensorInstances of DistArray representing multiple incoming 
    %    messages
    %    - Generate a random feature vector for each message with RFGEProdMap 
    %    Kronecker product all of them.
    %    - All internal non-DistNormal will be converted to converted by  moment 
    %    matching to DistNormal when computing the expected product kernel.
    %
    
    properties(SetAccess=private)
        % Gaussian width^2 for the embedding kernel for each incoming variable 
        % A numeric array
        gwidth2s;

        % number of total random features
        % If there are c incoming variables, e=floor(numFeatures^(1/c)) will be the number 
        % of features for each variable. The length of the final feature vector 
        % e^c (exponential in c!).
        numFeatures;

        % cell array of RFGEProdMap's. 
        % Maintain a map for each incoming message.
        eprodMaps;
    end
    
    methods
        function this=RFGProductEProdMap(gwidth2s, numFeatures)
            assert(all(gwidth2s>0));
            assert(numFeatures>0);
            this.gwidth2s=gwidth2s;
            % this.numFeatures may change later if numFeatures/c is not integer.
            this.numFeatures=numFeatures;
            this.eprodMaps={};
        end

        function Z=genFeatures(this, T)
            assert(isa(T, 'TensorInstances'));
            this.initMap(T);
            C=T.instancesCell;
            numVars=length(C);
            n=T.count();
            Zs=cell(1, numVars);
            for i=1:numVars
                % Each DistArray must have the same length
                assert(isa(C{i}, 'DistArray'), ['Each input in TensorInstances '...
                    'is expected to be a DistArray']);
                Zs{i}=this.eprodMaps{i}.genFeatures(C{i});
            end
            Z=Zs{1};
            for i=2:numVars
                % Kronecker product.
                Z=MatUtils.colKronecker(Z, Zs{i} );
            end
            assert(size(Z, 2)==n);
            assert(size(Z, 1)==this.numFeatures);
        end


        function M=genFeaturesDynamic(this, T)
            % Generate feature vectors in the form of DynamicMatrix.
            assert(isa(T, 'TensorInstances') );
            this.initMap(T);
            g=this.getGenerator(T);
            n=T.count();
            M=DefaultDynamicMatrix(g, this.numFeatures, n);
        end

        function g=getGenerator(this, T)
            g=@(I, J)this.generator(T, I, J);

        end

        function Z=generator(this, T, I, J )
            % I=indices of features, J=sample indices 
            assert(isa(T, 'TensorInstances') );
            this.initMap(T);
            C=T.instancesCell;
            c=T.tensorDim();
            nfEach=this.numFeatures^(1/c);
            assert(mod(nfEach, 1)==0); % this should be integer 

            % Indices needed from each feature map 
            mapI=cell(1, c);
            [mapI{:}]=ind2sub(nfEach*ones(1, c), I);
            % reverse mapI so that the last index changes first.
            mapI=mapI(end:-1:1);
            
            Z=1;
            for i=1:c
                % Each DistArray must have the same length
                assert(isa(C{i}, 'DistArray'), ['Each input in TensorInstances '...
                    'is expected to be a DistArray']);
                g=this.eprodMaps{i}.getGenerator(C{i});
                % This can be improved. mapI{i} may contain many duplicate indices.
                % We should not need to recompute. Just compute once and repmat it.
                % **** Improve later ***
                Z=Z.*g(mapI{i}, J);
            end
            assert(size(Z,1)==length(I));
            assert(size(Z,2)==length(J));

        end

        function fm=cloneParams(this, numFeatures)
            fm=RFGProductEProdMap(this.gwidth2s, numFeatures);
        end

        function s=shortSummary(this)
            s = sprintf('%s(gw2s=[%s])', ...
                mfilename, num2str(this.gwidth2s)) ;
        end
        
        % from PrimitiveSerializable interface
        function s=toStruct(this)
            s.className=class(this);
            % A list of width^2 
            s.gwidth2s=this.gwidth2s;
            s.numFeatures=this.numFeatures;
            % eprodMaps is a cell array of RFGEProdMap's
            maps = cell();
            for i=1:length(this.eprodMaps)
                maps{i} = this.eprodMaps{i}.toStruct();
            end
            s.eprodMaps=maps;
        end
    end
    
    methods(Access=private)
        function initMap(this, T)
            % Initialize the map only once. 
            if isempty(this.eprodMaps)
                assert(isa(T, 'TensorInstances'));
                c=T.tensorDim();
                maps=cell(1, c);
                nf=max(2, floor(this.numFeatures^(1/c)) );
                for i=1:c
                    maps{i}=RFGEProdMap(this.gwidth2s(i), nf);
                end
                this.eprodMaps=maps;
                % The total numFeatures <= the original numFeatures
                this.numFeatures=nf^c;

            end
        end
    end %end private methods
    
    methods(Static)

        function FMs = candidatesAvgCov(T, medf, numFeatures, subsamples )
            % - Generate a cell array of FeatureMap candidates from medf,
            % a list of factors to be  multiplied with the 
            % diagonal of the average covariance matrices.
            %
            % - subsamples can be used to limit the samples used to compute
            % the average
            %
            assert(isa(T, 'TensorInstances'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(all(medf>0));
            if nargin < 4
                subsamples = 5000;
            end
            numInput=T.tensorDim();
            meanVars=zeros(1, numInput);
            for i=1:numInput
                da=T.instancesCell{i};
                avgCov=RFGEProdMap.getAverageCovariance(da, subsamples);
                meanVars(i)=mean(diag(avgCov));
            end

            % total number of candidats = len(medf). Quite cheap.
            FMs = cell(1, length(medf));
            for ci=1:length(medf)
                gwidth2s=meanVars*medf(ci);
                map = RFGProductEProdMap(gwidth2s, numFeatures);
                FMs{ci} = map;
            end
        end %end candidates() method

    end %end static methods

end

