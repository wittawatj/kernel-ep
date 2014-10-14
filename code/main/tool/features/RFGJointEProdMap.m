classdef RFGJointEProdMap < FeatureMap & PrimitiveSerializable
    %RFGJOINTEPRODMAP Random Fourier features for expected product kernel 
    %using Gaussian kernel on joint mean embeddings. 
    %    - Input is a TensorInstances of DistArray representing multiple incoming 
    %    messages 
    %    - All non-DistNormal will be converted to converted by  moment 
    %    matching to DistNormal when computing the expected product kernel.
    %    - Merge all (converted) incoming messages into a big Gaussian distribution 
    %    and compute random features for expected product kernel of the big 
    %    Gaussians.
    %
    
    properties(SetAccess=protected)
        % Gaussian widths^2 for the embedding kernel for each incoming variable.
        % A numeric array. Length=number of incoming variables.
        % Reciprocal of gwidth2s are used in drawing W's.
        gwidth2s;
        
        % number of random features
        numFeatures;

        % a RFGEProdMap
        eprodMap;

    end
    
    methods
        function this=RFGJointEProdMap(gwidth2s, numFeatures)
            assert(all(gwidth2s>0));
            assert(numFeatures>0);
            this.gwidth2s=gwidth2s(:)';
            assert(mod(numFeatures, 1)==0);
            this.numFeatures=numFeatures;
            this.eprodMap=[];
        end

        function Z=genFeatures(this, T)
            % Z = numFeatures x n
            assert(isa(T, 'TensorInstances') ); 
            this.initMap(T);
            D=RFGJointEProdMap.tensorToJointGaussians(T);
            Z=this.eprodMap.genFeatures(D);

        end

        function M=genFeaturesDynamic(this, T)
            % Generate feature vectors in the form of DynamicMatrix.
            assert(isa(T, 'TensorInstances'));
            this.initMap(T);
            D=RFGJointEProdMap.tensorToJointGaussians(T);
            g=@(I, J)this.eprodMap.generator(D, I, J);
            n=length(T);
            assert(length(T)==length(D));
            M=DefaultDynamicMatrix(g, this.numFeatures, n);
        end

        function g=getGenerator(this, T)
            g=@(I, J)this.generator(T, I, J);

        end

        function Z=generator(this, T, I, J )
            % I=indices of features, J=sample indices 
            assert(isa(T, 'TensorInstances'));
            this.initMap(T);
            RT=T.instances(J);
            D=RFGJointEProdMap.tensorToJointGaussians(RT);
            Z=this.eprodMap.generator(D, I, 1:length(D));

        end

        function fm=cloneParams(this, numFeatures)
            fm=RFGJointEProdMap(this.gwidth2s, numFeatures);
            
        end

        function D=getNumFeatures(this)
            D = this.numFeatures;
        end

        function s=shortSummary(this)
            s = sprintf('%s(gw2s=[%s])', ...
                mfilename, num2str(this.gwidth2s)) ;
        end

        % from PrimitiveSerializable interface
        function s=toStruct(this)
            s.className=class(this);
            s.gwidth2s=this.gwidth2s;
            s.numFeatures=this.numFeatures;
            s.eprodMap=this.eprodMap.toStruct();
        end

    end %end methods

    methods(Access=private)
        function initMap(this, T)
            % Initialize the map only once. 
            if isempty(this.eprodMap)
                assert(isa(T, 'TensorInstances'));
                c=T.tensorDim();
                dimf=@(da)(unique(da.d));
                dims=cellfun(dimf, T.instancesCell);
                assert(length(this.gwidth2s)==c,['length of specified gwidth2s '...
                   'does not match tensorDim()' ]);
                totalDim=sum(dims);
                % stretch gwidth2s to have the same length as totalDim
                C=arrayfun(@(n, t)repmat(n, 1, t), this.gwidth2s, dims, ...
                    'UniformOutput', false);
                W=diag(1./sqrt([C{:}]))*randn(totalDim, this.numFeatures);
                B=rand(1, this.numFeatures)*2*pi;
                this.eprodMap=RFGEProdMap.createFromWeights(W, B);
            end
        end
    end

    methods(Static)
        function D=tensorToJointGaussians(T)
            % Convert a TensorInstances T of DistArray's into an array of DistNormal.
            % Each DistNormal is constructed by converting all incoming messages 
            % into a single Gaussian. The final covariance is given by stacking 
            % diagonally all covariances (block matrix). 
            % Converting another distribution is done by extracting mean, variance 
            % and constructing a Gaussian out of them. 
            %
            assert(isa(T, 'TensorInstances'));
            n=T.count();
            nvars=T.tensorDim();
            C=T.instancesCell;
            Dcell=cell(1, n);
            for j=1:n
                Ms=cell(1, nvars);
                % cell array of covariance matrices
                Vs=cell(1, nvars);
                for i=1:nvars
                    Dij=C{i}.get(j);
                    assert(isa(Dij, 'Distribution'));
                    Ms{i}=Dij.mean(:);
                    Vs{i}=Dij.variance;
                end
                % Stack all means and variances
                M=vertcat(Ms{:});
                V=blkdiag(Vs{:});
                Dcell{j}=DistNormal(M, V);
            end
            D=[Dcell{:}];
            assert(length(D)==n);
        end

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
                map = RFGJointEProdMap(gwidth2s, numFeatures);
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
            % list of dimensions
            dims=cellfun(@(da)unique([ da.d ]), T.instancesCell);
            assert(isnumeric(dims));

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
                % make gwidth2s the same size as numInput.
                C=arrayfun(@(n, t)repmat(n, 1, t), inputWidth2s, dims,...
                   'UniformOutput', false );
                gwidth2s=[C{:}];
                assert(length(gwidth2s)==sum(dims));
                map = RFGJointEProdMap(gwidth2s, numFeatures);
                FMs{ci} = map;
            end

        end %end candidates() method

    end % end static methods

    
end

