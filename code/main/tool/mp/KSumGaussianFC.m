classdef KSumGaussianFC < KernelFeatureFC
    %KSUMGAUSSIANFC A function classs defining sum of Gaussian kernels on features 
    %extracted using the specified feature extractor.
    %   .
    
    properties(SetAccess = protected)
    end
    
    methods
        % width w enters to kernel with 1/w^2 in the exp.
        function this = KSumGaussianFC(varargin)
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
            assert(size(F1sc, 1) == size(F2sc, 1));
            d = size(F1sc, 1);
            Kmat = zeros(size(F1sc, 2), size(F2sc, 2));
            for i=1:d
                f1i = F1sc(i, :);
                f2i = F2sc(i, :);

                D2 = bsxfun(@minus, f1i', f2i).^2;
                Kmat = Kmat + exp(-D2);
            end
        end

        function obj = finalize(this)
            % construct a dummy obj. Modify later.
            %obj = KSumGaussianFC(this.kerParams, this.featureExtractor, ...
            %    this.centerInstances.instances(1), this.inputInstances.instances(1));
            obj = KSumGaussianFC();
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

