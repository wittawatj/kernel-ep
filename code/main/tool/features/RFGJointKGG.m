classdef RFGJointKGG < FeatureMap & PrimitiveSerializable
    %RFGJOINTKGG Random Fourier features for Gaussian on mean embeddings
    %using Gaussian kernel (for mean embeddings) on joint distributions. 
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
        % Length=number of incoming variables.
        % One parameter vector for incoming message.
        % Reciprocal of gwidth2s are used in drawing W's.
        embed_width2s_cell;

        % width2 for the outer Gaussian kernel on the mean embeddings.
        % A scalar.
        outer_width2;
        
        % number of random features. This is the same as nfOut i.e., #features 
        % for the outer map.
        numFeatures;

        % a RFGEProdMap
        eprodMap;

        % number of features for the inner map (expected product kernel). An integer.
        innerNumFeatures;

        % Din x Dout  
        Wout;

        % A vector of length numFeatures.   Drawn from U[0, 2*pi]
        Bout;
    end
    
    methods
        function this=RFGJointKGG(embed_width2s_cell, outer_width2, ...
                innerNumFeatures, numFeatures)

            assert(isscalar(outer_width2));
            assert(outer_width2 > 0);
            assert(iscell(embed_width2s_cell));
            assert(innerNumFeatures > 0);
            assert(numFeatures>0);

            this.embed_width2s_cell = embed_width2s_cell;
            this.outer_width2 = outer_width2;
            this.numFeatures=numFeatures;
            this.innerNumFeatures = innerNumFeatures;
            this.eprodMap=[];
            this.Wout = [];
            this.Bout = [];
        end

        function Z=genFeatures(this, T)
            % Z = numFeatures x n
            assert(isa(T, 'TensorInstances') ); 
            this.initMap(T);
            Z = this.generator(T, 1:this.numFeatures, 1:length(T));
        end

        function M=genFeaturesDynamic(this, T)
            % Generate feature vectors in the form of DynamicMatrix.
            assert(isa(T, 'TensorInstances'));
            this.initMap(T);
            D=RFGJointEProdMap.tensorToJointGaussians(T);
            g=@(I, J)this.generator_dist(D, I, J);
            n=length(T);
            assert(length(T)==length(D));
            M=DefaultDynamicMatrix(g, this.numFeatures, n);
        end

        function g=getGenerator(this, T)
            g=@(I, J)this.generator(T, I, J);

        end

        function Zout=generator(this, T, I, J )
            % I=indices of features, J=sample indices 
            assert(isa(T, 'TensorInstances'));
            this.initMap(T);
            RT=T.instances(J);
            D=RFGJointEProdMap.tensorToJointGaussians(RT);
            Zout = this.generator_dist(D, I, 1:length(D));
        end

        function Zout = generator_dist(this, D, I, J)
            assert(isa(D, 'Distribution'));
            D = D(J);
            % TODO: This part is not memory efficient. Form a potentially big matrix 
            % here. Fix it.
            Zin=this.eprodMap.generator(D, 1:this.innerNumFeatures, 1:length(D));
            assert(size(Zin, 1) == this.innerNumFeatures);
            WT = this.Wout';
            subW = WT(I, :);
            subB = this.Bout(I)';
            Zout = cos(bsxfun(@plus, subW*Zin, subB))*sqrt(2/this.numFeatures);
        end

        function fm=cloneParams(this, numFeatures, innerNumFeatures)
            fm=RFGJointKGG(this.embed_width2s_cell, this.outer_width2, ...
                innerNumFeatures, numFeatures);
        end

        function D=getNumFeatures(this)
            D = this.numFeatures;
        end

        function s=shortSummary(this)
            s = sprintf('%s(embed_w2=[%s], outer_w2=%.2f)', ...
                mfilename, num2str(MatUtils.flattenCell(this.embed_width2s_cell)), ...
                this.outer_width2) ;
        end

        % from PrimitiveSerializable interface
        function s=toStruct(this)
            s.className=class(this);
            % make it a row cell array
            s.embed_width2s_cell = {this.embed_width2s_cell{:}};
            s.outer_width2 = this.outer_width2;
            s.numFeatures=this.numFeatures;
            s.innerNumFeatures = this.innerNumFeatures;
            s.eprodMap=this.eprodMap.toStruct();
            s.Wout = this.Wout;
            s.Bout = this.Bout;
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
                assert(length(this.embed_width2s_cell)==c, ['length of specified parameters '...
                   'does not match tensorDim()' ]);
                totalDim=sum(dims);

                % flatten embed_width2s_cell
                flat_embed2s = MatUtils.flattenCell(this.embed_width2s_cell);
                Win = diag(1./sqrt(flat_embed2s))*randn(totalDim, this.innerNumFeatures);
                Bin=rand(1, this.innerNumFeatures)*2*pi;
                this.eprodMap=RFGEProdMap.createFromWeights(Win, Bin);

                % For outer kernel
                this.Wout = randn(this.innerNumFeatures, this.numFeatures)/sqrt(this.outer_width2);
                this.Bout = rand(1, this.numFeatures)*2*pi;
            end
        end
    end

    methods(Static)
        function FMs_cell = candidatesAvgCov(T, medf, innerNumFeatures, numFeatures, subsamples )
            % - Generate a cell array of FeatureMap candidates from medf,
            % a list of factors to be  multiplied with the 
            % diagonal of the average covariance matrices.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            %
            assert(innerNumFeatures > 0);
            assert(numFeatures > 0);
            assert(isa(T, 'TensorInstances'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(all(medf>0));
            if nargin < 5
                subsamples = min(4000, length(T));
            end
            %numInput=T.tensorDim();
            jointGauss = KGGaussianJoint.toJointDistArray(T.instancesCell);
            % Dimension of the joint Gaussians.
            dim = unique([jointGauss.d]);
            assert(length(dim)==1);
            % Average covariance as a matrix.
            avgCov=RFGEProdMap.getAverageCovariance(jointGauss, subsamples);
            % Gaussian width-squared 
            % The average covariance will be multipled with median factor at the 
            % end.
            %
            embed_width2 = diag(avgCov);
            dims=cellfun(@(da)unique([ da.d ]), T.instancesCell);
            FMs_cell = RFGJointKGG.candidates(jointGauss, dims, {embed_width2}, medf, ...
                innerNumFeatures, numFeatures, subsamples);
        end 
        
        function FMs_cell = candidates(D, dims, flatten_embed_width2s_cell, medf, innerNumFeatures, ...
                numFeatures, subsamples )
            % - Generate a cell array of FeatureMap candidates from medf,
            % a list of factors to be  multiplied with the 
            % median heuristic.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            %

            assert(isa(D, 'Distribution'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(innerNumFeatures > 0);
            assert(numFeatures > 0);
            assert(iscell(flatten_embed_width2s_cell));
            assert(~isempty(flatten_embed_width2s_cell));
            assert(all(medf>0));
            if nargin < 6
                subsamples = 1500;
            end

            dim = unique([D.d]);

            FMs = cell(length(flatten_embed_width2s_cell), length(medf));
            for i=1:length(flatten_embed_width2s_cell)
                ewidth = flatten_embed_width2s_cell{i};
                assert(length(ewidth) == dim);
                [~, med]= KGGaussian.compute_meddistances(D, {ewidth}, subsamples);
                for j=1:length(medf)
                    w2 = medf(j)*med;
                    embed_width2s_cell = MatUtils.partitionArray(ewidth, dims);
                    FMs{i, j} = RFGJointKGG(embed_width2s_cell, w2, innerNumFeatures, ...
                        numFeatures);
                end
            end
            FMs_cell = {FMs{:}};

        end %end candidates() method

    end % end static methods

    
end

