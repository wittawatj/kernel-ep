classdef KGaussianFC < KernelFeatureFC
    %KGAUSSIANFC A function class defining Gaussian kernels on features extracted
    % using the specified feature extractor.    
    % %   .
    
    properties(SetAccess = protected)
    end
    
    methods
        function this = KGaussianFC(varargin)
            v = varargin;
            this@KernelFeatureFC(v{:});

        end

        % implementing abstract method
        function Kmat = kernelEval(this, F1, F2)
            if isempty(F1) || isempty(F2)
                Kmat = [];
                return;
            end
            lwidths = this.kerParams;
            F1sc = diag(1./lwidths)*F1;
            F2sc = diag(1./lwidths)*F2;
            D = bsxfun(@plus, sum(F1sc.^2, 1)', sum(F2sc.^2, 1)) - 2*F1sc'*F2sc;
            Kmat = exp(-D);
        end

        function obj = finalize(this)
            % construct a dummy obj. Modify later.
            %obj = KGaussianFC(this.kerParams, this.featureExtractor, ...
            %    this.centerInstances.instances(1), this.inputInstances.instances(1));
            obj = KGaussianFC();
            obj.options = this.options;
            obj.featureExtractor = this.featureExtractor;
            obj.inputFeatures = [];
            obj.centerFeatures = this.centerFeatures(:, this.markedInd);
            obj.markedInd = 1:size(obj.centerFeatures, 2);
            obj.weightMat = this.weightMat;
            obj.kerParams = this.kerParams;
        end
    end % end methods
end

