classdef RandFourierGaussMVMap < FeatureMap & PrimitiveSerializable
    %RANDFOURIERGAUSSMVMAP RandFourierGaussMap for Distribution using its mean
    %and variance as input.    
    %   - Input to genFeatures() is expected to be a  DistArray
    %   or TensorInstances of DistArray's

    properties (SetAccess=private)
        % Gaussian width^2 for means 
        % A numeric array of the form (mwidth2_1,  mwidth2_2,  ...), one 
        % parameter for each input variable to the factor.
        mwidth2s;
        % Gaussian width^2 for variances 
        % A numeric array of the form (vwidth2_1,  vwidth2_2,  ...)
        vwidth2s;
        % number of features to generate
        numFeatures;

        % underlying RandFourierGaussMap. 
        % There is only one map for the whole tensor input.
        % Generated the first time this map is used.
        rfgMap;
    end

    methods

        function this=RandFourierGaussMVMap(mwidth2s, vwidth2s, numFeatures )
            assert(numFeatures > 0);
            this.numFeatures = numFeatures;
            % If the input is a DistArray, then mwidth2s and vwidth2s contain 1 element. 
            % If the input is a TensorInstances, then 
            % mwidth2s = (mwidth2_1, mwidth2_2, ...) 
            % vwidth2s = (vwidth2_1, vwidth2_2, ...)
            % Length = tensorDim()
            assert(~isempty(mwidth2s));
            assert(~isempty(vwidth2s));
            this.mwidth2s = mwidth2s(:)';
            this.vwidth2s = vwidth2s(:)';
        end

        function Z=genFeatures(this, X)  
            % X is a Distribution or DistArray or TensorInstances of DistArray
            % Z = numFeatures x n

            this.initMap(X);
            % In = stack of rescaled means and variances
            In = this.toMVStack(X);
            assert(isnumeric(In));
            Z = this.rfgMap.genFeatures(In);

        end

        function g=getGenerator(this, X)
            this.initMap(X);
            % In = stack of rescaled means and variances
            In = this.toMVStack(X);
            g = this.rfgMap.getGenerator(In);

        end

        function M=genFeaturesDynamic(this, X)
            assert(isa(X, 'DistArray') || isa(X, 'TensorInstances'));
            g=this.getGenerator(X);
            n=X.count();
            M=DefaultDynamicMatrix(g, this.numFeatures, n);
        end

        function s=shortSummary(this)
            s = sprintf('RandFourierGaussMVMap(mw2s=[%s], vw2s=[%s])', ...
                num2str(this.mwidth2s), num2str(this.vwidth2s) );
        end

        function map = cloneParams(this, numFeatures)
            % Clone this RandFourierGaussMVMap and keep the same mwidth2s 
            % and vwidth2s parameters while accepting a new numFeatures.
            % The internal weight vector will change due to randomness.
            %
            map = RandFourierGaussMVMap(this.mwidth2s, ...
                this.vwidth2s, numFeatures);
        end

        function D=getNumFeatures(this)
            D = this.numFeatures;
        end

        function s=saveobj(this)
            s.mwidth2s=this.mwidth2s;
            s.vwidth2s=this.vwidth2s;
            s.numFeatures=this.numFeatures;
            s.rfgMap=this.rfgMap;
        end

        % from PrimitiveSerializable interface
        function s=toStruct(this)
            s.className=class(this);
            % A list of width^2 for mean. One parameter for each input.
            s.mwidth2s=this.mwidth2s;
            % A list of width^2 for variance. One parameter for each input.
            s.vwidth2s=this.vwidth2s;
            s.numFeatures=this.numFeatures;
            s.rfgMap=this.rfgMap.toStruct();
        end
    end

    methods(Access=private)
        function initMap(this, X)
            % Initialize the map only once. 
            if isempty(this.rfgMap)
                dim = RandFourierGaussMVMap.getTotalDim(X);
                nf = this.numFeatures;
                % Set width to 1 because we will rescale the input here.
                this.rfgMap = RandFourierGaussMap(1, nf, dim);
            end
        end

        function [In, Ms, Vs] = toMVStack(this, X)
            % X is a DistArray or TensorInstances of DistArray
            % Z = numFeatures x n
            assert(isa(X, 'DistArray') || isa(X, 'TensorInstances'));
            [Ms, Vs]=RandFourierGaussMVMap.getAllMV(X);

            % rescale the inputs
            CM = cellfun(@(M, p)M/sqrt(p), Ms, num2cell(this.mwidth2s), ...
                'UniformOutput', false);
            CV = cellfun(@(V, p)V/sqrt(p), Vs, num2cell(this.vwidth2s), ...
                'UniformOutput', false);
            SM = vertcat(CM{:});
            SV = vertcat(CV{:});

            In = [SM; SV];
        end


    end %end private methods

    methods (Static)

        function obj=loadobj(s)
            obj=RandFourierGaussMVMap(s.mwidth2s, s.vwidth2s, s.numFeatures );
            obj.rfgMap=s.rfgMap;
        end

        function [Means, Vars]=getMV( X)
            % Get mean and variance from a DistArray
            assert(isa(X, 'DistArray'));
            sq = size(X.variance, 1);
            if sq > 1
                % multivariate Gaussian
                Vars = reshape(X.variance, sq^2, length(X));
            else
                % univariate Gaussian. [X.variance] is a row vector.
                Vars = [X.variance];
            end
            Means = [X.mean];
        end

        function dim=getTotalDim(X)
            assert(X.count()>0, 'X is empty');
            x = X.instances(1);
            [Ms, Vs]=RandFourierGaussMVMap.getAllMV(x);
            dms = cellfun(@(M)size(M,1), Ms );
            dvs = cellfun(@(V)size(V,1), Vs);
            dim = sum(dms) + sum(dvs);
            
        end

        function [Ms, Vs]=getAllMV(X)
            % Get all means and variances in a cell array
            if isa(X, 'DistArray')
                [Means, Vars] = RandFourierGaussMVMap.getMV(X); 
                Ms = {Means};
                Vs = {Vars};
            elseif isa(X, 'TensorInstances')
                % Cell array of DistArray's
                C = X.instancesCell;
                Ms = cell(1, length(C));
                Vs = cell(1, length(C));
                for i=1:length(C)
                    assert(isa(C{i}, 'DistArray'), ...
                        'Each input in TensorInstances is expected to be a DistArray');
                    [M, V] = RandFourierGaussMVMap.getMV(C{i});
                    Ms{i} = M;
                    Vs{i} = V;
                end
            else
                error('Invalid input type to %s', mfilename);
            end

        end

        function FMs = candidatesFlatMedf(X, medf, numFeatures, subsamples )
            % - Generate a cell array of FeatureMap candidates from medf,
            % a list of factors to be  multiplied with the 
            % median of all means, variances.
            % - Steps
            %  1. compute median of means mmeds of all incoming variables
            %  2. compute median of variances vmeds of all incoming variables
            %  3. Generate (mmeds, vmeds)*medf(i) for all i
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            %
            assert(isa(X, 'DistArray') || isa(X, 'TensorInstances'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(all(medf>0));
            if nargin < 4
                subsamples = 2000;
            end
            [Ms, Vs] = RandFourierGaussMVMap.getAllMV(X);
            n = size(Ms{1}, 2);
            numInput = length(Ms);
            % mmeds contains median for each input
            mmeds = zeros(1, numInput);
            vmeds = zeros(1, numInput);
            I = randperm(n, min(n, subsamples));
            for i=1:numInput
                M = Ms{i};
                V = Vs{i};

                mmeds(i) = meddistance(M(:, I))^2;
                vmeds(i) = meddistance(V(:, I))^2;
            end

            % total number of candidats = len(medf). Quite cheap.
            FMs = cell(1, length(medf));
            for ci=1:length(medf)
                map=RandFourierGaussMVMap(mmeds*medf(ci), vmeds*medf(ci), numFeatures);
                FMs{ci} = map;
            end

        end %end candidates() method

        function FMs = candidates(X, mean_medf, var_medf, numFeatures, subsamples )
            % - Generate a cell array of FeatureMap candidates from a list of
            % mean_medf, a list of factors to be  multipled with the 
            % pairwise median distance of the means.
            % - Same semantic for var_medf but for the variance.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            % - This method is analogous to candidates() as in KMVGauss1.
            assert(isa(X, 'DistArray') || isa(X, 'TensorInstances'));
            [Ms, Vs] = RandFourierGaussMVMap.getAllMV(X);
            n = size(Ms{1}, 2);
            numInput = length(Ms);
            if nargin < 5
                subsamples = 1500;
            end
            assert(isnumeric(mean_medf));
            assert(~isempty(mean_medf));
            assert(isnumeric(var_medf));
            assert(~isempty(var_medf));
            assert(all(mean_medf > 0));
            assert(all(var_medf > 0));

            % mmeds contains median for each input
            mmeds = zeros(1, numInput);
            vmeds = zeros(1, numInput);
            I = randperm(n, min(n, subsamples));
            for i=1:numInput
                M = Ms{i};
                V = Vs{i};

                mmeds(i) = meddistance(M(:, I))^2;
                vmeds(i) = meddistance(V(:, I))^2;
            end

            % each index in mvInd identifies one combination of mean_medf
            % and var_medf.
            mvSize = length(mean_medf)*length(var_medf);
            % total number of candidats = [len(mean_medf)*len(var_medf)]^num_input where num_input = tensor dimension of X
            % Total combinations can be huge ! Be careful. Exponential in the 
            % number of inputs
            totalComb = mvSize^numInput;
            FMs = cell(1, totalComb);
            % temporary vector containing indices
            I = cell(1, numInput);
            for ci=1:totalComb
                [I{:}] = ind2sub( mvSize*ones(1, numInput), ci);
                mwidth2s = zeros(1, numInput);
                vwidth2s = zeros(1, numInput);
                for i=1:length(I)
                    [mi, vi] = ind2sub([length(mean_medf), ...
                        length(var_medf)], I{i} );
                    mwidth2s(i) = mmeds(i)*mean_medf(mi);
                    vwidth2s(i) = vmeds(i)*var_medf(vi);
                end
                % Construct a RandFourierGaussMVMap candidate
                map = RandFourierGaussMVMap(mwidth2s, vwidth2s, numFeatures);
                FMs{ci} = map;
            end

        end %end candidates() method

    end % end static methods
end

