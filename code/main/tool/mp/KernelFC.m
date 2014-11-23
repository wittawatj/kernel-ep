classdef KernelFC < MPFunctionClass
    %KERNELFC A function class based on kernel functions on data instances.
    %   - No FeatureExtractor as in KernelFeatureFC
    %   - Content is highly similar to KernelFeatureFC. ** Code factorize later ? **

    properties(SetAccess = protected)
        % An instance of Options
        options;

        centerInstances;
        % input samples. Instances. n samples.
        % Inherited
        %inputInstances;

        % indices (not 0-1) indicating which centers are marked to be included 
        % in the final function.
        markedInd;

        % W matrix where each column is one w_i of length dim(output)
        weightMat = [];

        % Kernel object. Must work on centerInstances, inputInstances.
        % If inputInstances is TensorInstances, the kernel is likely to be a 
        % KProduct.
        kernel;
    end

    methods
        function this = KernelFC(centerInstances, inputInstances, kernel)
            %KernelFC(centerInstances, inputInstances)
                
            % fe = a FeatureExtractor
            if nargin <= 0
                % private constructor.
                return;
            end
            %in = varargin;
            %centerInstances = in{1};
            %inputInstances = in{2};

            assert(isa(inputInstances, 'Instances'));
            assert(isa(centerInstances, 'Instances'));
            assert(isa(kernel, 'Kernel'));
            this.centerInstances = centerInstances;
            this.inputInstances = inputInstances;
            this.kernel = kernel;

            this.markedInd = [];
            this.options = this.getDefaultOptions();
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';
            kv.mp_subsample = ['Number of samples to consider in every iteration.', ...
            ' Random subsampling will be used.'];
            kv.mp_basis_subsample = ['Number of kernel centers (basis) to consider in ',...
            'every iteration. Random subsampling.'];

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.mp_subsample = inf(1);
            st.mp_basis_subsample = inf(1);

            Op=Options(st);
        end

        % evaluate kernel on the data. X1, X2 are Instances.
        function KMat = kernelEval(this, X1, X2)
            assert(isa(X1, 'Instances'));
            assert(isa(X2, 'Instances'));
            ker = this.kernel;
            KMat = ker.eval(X1.getAll(), X2.getAll());
        end

        function [crossRes, G, wt, searchMemento] = findBestBasisFunction(this, R, regParam)
            C = this.centerInstances;
            totalBasis = length(C);
            if isempty(C) || length(this.markedInd) == totalBasis
                % If all candidates are selected,...
                % no more candidates 
                crossRes = -inf(1);
                G = [];
                searchMemento = [];
                return;
            end
            n = length(this.inputInstances);
            subsample = this.opt('mp_subsample');
            If = randperm(n, min(subsample, n));
            % Exclude already selected basis functions
            I = setdiff(1:totalBasis, this.markedInd);
            % Subsample basis functions
            basis_subsample = this.opt('mp_basis_subsample');
            I = I(randperm(length(I), min(basis_subsample, length(I))));
            Kmat = this.evaluateOnTrainingCenterInd(I, If);

            % row vector. same length as the #center instances
            z = sum(Kmat.*Kmat, 2)' + regParam;
            % cross correlations on subset of inputInstances
            Cross = sum( (R(:, If)*Kmat').^2, 1) ./ z;
            % pick best basis function g by taking max cross correlation
            [crossRes, mi] = max(Cross);
            gind = I(mi);

            % Evaluate the function on full data set 
            G = this.evaluateOnTrainingCenterInd(gind, 1:n);
            % compute w
            wt = R*G'/(G*G' + regParam);
            % search memento for marking the selected function later 
            mem = struct();
            mem.centerIndex = gind;
            mem.wt = wt;
            searchMemento = mem;
        end

        function markBestBasisFunction(this, searchMemento)
            assert(isstruct(searchMemento));
            cind = searchMemento.centerIndex;
            assert(~ismember(cind, this.markedInd));
            % This will affect next findBestBasisFunction() and evaluate()
            this.markedInd = [this.markedInd, cind];
            wt = searchMemento.wt;
            this.weightMat = [this.weightMat, wt];
        end

        function G = evaluateOnTraining(this)
            % return: G (#marked x inputInstances count)
            n = length(this.inputInstances);
            G = this.evaluateOnTrainingCenterInd(this.markedInd, 1:n);

        end

        function G = evaluateOnTrainingSubset(this, Ind)
            % return: G (#marked x length(Ind ) )
            G = this.evaluateOnTrainingCenterInd(this.markedInd, Ind);

        end

        function setWeightMatrix(this, W)
            assert(length(this.markedInd) == size(W, 2));
            this.weightMat = W;
        end

        function c = countSelectedBases(this)
            c = length(this.markedInd);
            assert(length(this.markedInd) == length(unique(this.markedInd)));
        end

        function Func = evalFunction(this, X)
            if isempty(this.markedInd)
                Func = [];
                return;
            end
            assert(isa(X, 'Instances'));
            C = this.centerInstances;
            Kmat = this.kernelEval(C.instances(this.markedInd), X);
            W = this.weightMat;
            Func = W*Kmat;
        end

        function G = evaluate(this, X)
            % return #marked x sample size 
            assert(isa(X, 'Instances'));
            C = this.centerInstances;
            Kmat = this.kernelEval(C.instances(this.markedInd), X);
            G = Kmat;
        end


        function G = evaluateOnTrainingCenterInd(this, centerInd, sampleInd)
            if isempty(centerInd) || isempty(sampleInd)
                G = [];
                return;
            end
            X = this.inputInstances;
            Xsub = X.instances(sampleInd);
            C = this.centerInstances;
            Csub = C.instances(centerInd);
            Kmat = this.kernelEval( Csub, Xsub);
            assert(all(size(Kmat) == [length(centerInd), length(sampleInd)]));
            G = Kmat;
        end

        function obj = finalize(this)
            % construct a dummy obj. Modify later.
            obj = KernelFC();
            C = this.centerInstances;
            obj.options = this.options;
            obj.centerInstances = C.instances(this.markedInd);
            % inputInstances not needed for evaluation. Only centerInstances needed.
            obj.inputInstances = [];
            obj.markedInd = 1:size(obj.centerInstances, 2);
            obj.weightMat = this.weightMat;
            obj.kernel = this.kernel;
            assert(size(obj.weightMat, 2) == length(obj.markedInd));
        end
    end % end methods
    
end

