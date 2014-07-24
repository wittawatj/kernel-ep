classdef RFGJointEProdMap < FeatureMap
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
            this.gwidth2s=gwidth2s;
            assert(mod(numFeatures, 1)==0);
            this.numFeatures=numFeatures;
            this.eprodMap=[];
        end

        function Z=genFeatures(this, T)
            % Z = numFeatures x n
            assert(isa(T, 'TensorInstances'));
            this.initMap(T);
            D=RFGJointEProdMap.tensorToJointGaussians(T);
            Z=this.eprodMap.genFeatures(D);

        end

        function M=genFeaturesDynamic(this, T)
            % Generate feature vectors in the form of DynamicMatrix.
            assert(isa(T, 'TensorInstances'));
            g=this.getGenerator(T);
            n=length(T);
            M=DefaultDynamicMatrix(g, this.numFeatures, n);
        end

        function g=getGenerator(this, T)
            g=@(I, J)this.generator(T, I, J);

        end

        function Z=generator(this, T, I, J )
            % I=indices of features, J=sample indices 
            assert(isa(T, 'TensorInstances'));
            this.initMap(T);
            D=RFGJointEProdMap.tensorToJointGaussians(T);
            Z=this.eprodMap.generator(D, I, J);

        end

        function s=shortSummary(this)
            s = sprintf('%s(mw2s=[%s])', ...
                mfilename, num2str(this.gwidth2s)) ;
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
                   'does not match tensomDim()' ]);
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
            % diagonally all covariances (same as stacking all precision matrices).
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
                    Dji=C{i}.get(j);
                    assert(isa(Dji, 'Distribution'));
                    Ms{i}=Dji.mean(:);
                    Vs{i}=Dji.variance;
                end
                % Stack all means and variances
                M=vertcat(Ms{:});
                V=blkdiag(Vs{:});
                Dcell{j}=DistNormal(M, V);
            end
            D=[Dcell{:}];
            assert(length(D)==n);
        end
    end % end static methods

    
end

