classdef KernelFeatureFC < MPFunctionClass
    %KERNELFEATUREFC A function class defining kernels on extracted features 
    %as basis functions.
    %   .
    
    properties(SetAccess = protected)
        % An instance of Options
        options;

        % input samples. Instances. n samples.
        % Inherited
        %inputInstances;

        featureExtractor;

        % features extracted from the samples. 
        % This can be big as samples may be big.
        % #features x n
        inputFeatures;

        % center samples.
        %centerInstances;
        % #features x #total basis (number of total basis functions)
        centerFeatures;

        % indices (not 0-1) indicating which centers are marked to be included 
        % in the final function.
        markedInd;
        % W matrix where each column is one w_i of length dim(output)
        weightMat = [];

    end
    
    methods(Abstract)
        % evaluate kernel on the extracted features F1 and F2
        KMat = kernelEval(this, F1, F2);
    end

    methods

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

        function [crossRes, G, wt, searchMemento] = findBestBasisFunction(this, R, regParam)
            C = this.centerFeatures;
            
            % R is dim(output) x #samples 
            if isempty(C) || length(this.markedInd) == size(this.centerFeatures, 2)
                % If all candidates are selected,...
                % no more candidates 
                crossRes = -inf(1);
                G = [];
                searchMemento = [];
                return;
            end
            n = size(this.inputFeatures, 2);
            subsample = this.opt('mp_subsample');
            If = randperm(n, min(subsample, n));
            totalBasis = size(this.centerFeatures, 2);
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
            assert(isa(X, 'Instances'));
            F = this.featureExtractor.extractFeatures(X);
            C = this.centerFeatures;
            Kmat = this.kernelEval(C(:, this.markedInd), F);
            W = this.weightMat;
            Func = W*Kmat;
        end

        function G = evaluate(this, X)
            % return #marked x sample size 
            Fx = this.featureExtractor.extractFeatures(X);
            C = this.centerFeatures;
            Kmat = this.kernelEval(C(:, this.markedInd), Fx);
            G = Kmat;
        end


        function G = evaluateOnTrainingCenterInd(this, centerInd, sampleInd)
            F = this.inputFeatures;
            C = this.centerFeatures;
            Kmat = this.kernelEval( C(:, centerInd), F(:, sampleInd));
            assert(all(size(Kmat) == [length(centerInd), length(sampleInd)]));
            G = Kmat;
        end
    end % end methods
    
end

