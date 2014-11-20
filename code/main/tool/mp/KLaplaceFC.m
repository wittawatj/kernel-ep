classdef KLaplaceFC < KernelFeatureFC
    %KLAPLACEFC A function class defining Laplace kernels on features extracted
    % using the specified feature extractor.    
    % %   .
    
    properties(SetAccess = protected)

    end
    
    methods
        function this = KLaplaceFC(varargin)
            v = varargin;
            this@KernelFeatureFC(v{:});
        end

        % implementing abstract method
        function Kmat = kernelEval(this, F1, F2)
            if isempty(F1) || isempty(F2)
                Kmat = [];
                return;
            end
            % expect a vector of widths, one for each dimension.
            lwidths = this.kerParams;
            F1sc = diag(1./lwidths)*F1;
            F2sc = diag(1./lwidths)*F2;
            D = sqrt(bsxfun(@plus, sum(F1sc.^2, 1)', sum(F2sc.^2, 1)) - 2*F1sc'*F2sc);
            Kmat = exp(-D);
        end

        function obj = finalize(this)
            % construct a dummy obj. Modify later.
            obj = KLaplaceFC(this.kerParams, this.featureExtractor, ...
                this.centerInstances.instances(1), this.inputInstances.instances(1));
            obj.inputInstances = [];
            obj.inputFeatures = [];
            obj.centerFeatures = this.centerFeatures(:, this.markedInd);
            obj.markedInd = 1:size(obj.centerFeatures, 2);
        end

    end
    
    methods(Access=private)
    end
    
end

