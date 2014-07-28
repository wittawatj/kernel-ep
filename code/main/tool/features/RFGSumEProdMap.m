classdef RFGSumEProdMap < FeatureMap
    %RFGSUMEPRODMAP Random Fourier features for sum of expected product kernel 
    %using Gaussian kernel for mean embeddings
    %    - Input is a TensorInstances of DistArray representing multiple incoming 
    %    messages
    %    - Generate a random feature vector for each message with RFGEProdMap and 
    %    stack all of them.
    %    - All internal non-DistNormal will be converted to converted by  moment 
    %    matching to DistNormal when computing the expected product kernel.
    %    
    
    properties(SetAccess=private)
        % Gaussian width^2 for the embedding kernel for each incoming variable 
        % A numeric array
        gwidth2s;

        % number of total random features
        % If there are c incoming variables, floor(numFeatures/c) will be the number 
        % of features for each variable. 
        numFeatures;

        % cell array of RFGSumEProdMap's. 
        % Maintain a map for each incoming message.
        eprodMaps;
    end
    
    methods
        function this=RFGSumEProdMap(gwidth2s, numFeatures)
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
            n=T.count();
            Zs=cell(1, length(C));
            for i=1:length(C)
                % Each DistArray must have the same length
                assert(isa(C{i}, 'DistArray'), ['Each input in TensorInstances '...
                    'is expected to be a DistArray']);
                Zs{i}=this.eprodMaps{i}.genFeatures(C{i});
            end
            % This make take a lot of memory
            Z=vertcat(Zs{:});
            assert(size(Z, 2)==n);
            assert(size(Z, 1)==this.numFeatures);
        end


        function M=genFeaturesDynamic(this, T)
            % Generate feature vectors in the form of DynamicMatrix.
            assert(isa(T, 'TensorInstances') );
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
            % For simplicity, require I, J to be non-decreasing.
            assert(issorted(I));
            assert(issorted(J));
            c=T.tensorDim();
            I=I(:);
            % MInd: same length as I indicating the feature map index to use 
            % e.g., MInd=(2,2,2,...,3,3,3,...)
            nfEach=this.numFeatures/c; % this should be integer 
            assert(mod(nfEach, 1)==0);
            MInd=ceil(I/nfEach);
            assert(issorted(MInd));
            assert(all(MInd>=1&MInd<=c));
            % U=(2, 3, ..).
            U=unique(MInd);
            Zs=cell(1, length(U));
            for i=1:length(U)
                unI=I(MInd==U(i));
                % normalize indices to be in [1, numFeatures/c]
                norI=unI-(U(i)-1)*nfEach;
                assert(all(norI>=1 & norI<=nfEach))
                DUi=T.instancesCell{U(i)};
                ZUi=this.eprodMaps{U(i)}.generator(DUi, norI, J);
                Zs{i}=ZUi;
            end
            Z=vertcat(Zs{:});
            assert(size(Z,1)==length(I));
            assert(size(Z,2)==length(J));

        end

        function fm=cloneParams(this, numFeatures)
            fm=RFGSumEProdMap(this.gwidth2s, numFeatures);
        end

        function s=shortSummary(this)
            s = sprintf('%s(gw2s=[%s])', ...
                mfilename, num2str(this.gwidth2s)) ;
        end
    end
    
    methods(Access=private)
        function initMap(this, T)
            % Initialize the map only once. 
            if isempty(this.eprodMaps)
                assert(isa(T, 'TensorInstances'));
                c=T.tensorDim();
                maps=cell(1, c);
                nf=floor(this.numFeatures/c);
                for i=1:c
                    maps{i}=RFGEProdMap(this.gwidth2s(i), nf);
                end
                this.eprodMaps=maps;
                % The total numFeatures <= the original numFeatures
                this.numFeatures=nf*c;

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
            % median distance.
            %
            assert(isa(T, 'TensorInstances'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(all(medf>0));
            if nargin < 4
                subsamples = 5000;
            end
            numInput=T.tensorDim();

            % median heuristics for each input variables
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
                map = RFGSumEProdMap(gwidth2s, numFeatures);
                FMs{ci} = map;
            end
        end %end candidates() method

        function FMs = candidates(T, medf, numFeatures, subsamples )
            % - Generate a cell array of FeatureMap candidates from medf,
            % a list of factors to be  multiplied with the 
            % median heuristic.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            %

            assert(isa(T, 'TensorInstances'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(all(medf>0));
            if nargin < 4
                subsamples = 1500;
            end
            numInput=T.tensorDim();
            % median heuristics for each input variables
            baseMeds=zeros(1, numInput);
            for i=1:numInput
                da=T.instancesCell{i};
                baseMeds(i)=RFGEProdMap.getBaseMedianHeuristic(da, subsamples);
            end

            % total number of candidats = len(medf)^numInput
            % Total combinations can be huge ! Be careful. Exponential in the 
            % number of inputs
            totalComb = length(medf)^numInput;
            FMs = cell(1, totalComb);
            % temporary vector containing indices
            I = cell(1, numInput);
            for ci=1:totalComb
                [I{:}] = ind2sub( length(medf)*ones(1, numInput), ci);
                II=cell2mat(I);
                inputWidth2s= medf(II).*baseMeds;
                map = RFGSumEProdMap(inputWidth2s, numFeatures);
                FMs{ci} = map;
            end

        end %end candidates() method

    end %end static methods
end

