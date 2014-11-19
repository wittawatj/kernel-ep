classdef KLaplaceFC < KernelFeatureFC
    %KLAPLACEFC A function class defining Laplace kernels on parameters of 
    %%incoming messages.
    %   .
    
    properties(SetAccess = protected)

        % widths matrix. Each column is to be used with one Laplace kernel.
        %widthsMat
        widths;

    end
    
    methods
        function this = KLaplaceFC(varargin)
            %KLaplaceFC(widths, fe, centerInstances, inputInstances)
            %KLaplaceFC(widths, fe, centerInstances, inputInstances, ...
            %centerFeatures, inputFeatures)
                
            % widths = parameters as in KLaplace
            % fe = a FeatureExtractor
            % laplaceCenters = instances to be used as centers for the kernel
            in = varargin;
            widths = in{1};
            fe = in{2};
            centerInstances = in{3};
            inputInstances = in{4};

            assert(isnumeric(widths));
            assert(all(widths > 0));
            assert(isa(fe, 'FeatureExtractor'));
            assert(isa(inputInstances, 'Instances'));
            assert(isa(centerInstances, 'Instances'));
            this.widths = widths;
            this.featureExtractor = fe;
            this.inputInstances = inputInstances;
            this.markedInd = [];
            %this.centerInstances = centerInstances;
            % cached features on all samples.
            %
            % Users have to make sure that the specified FeatureExtractor is 
            % compatible with the samples.

            if nargin <= 4
                
                %KLaplaceFC(widths, fe, centerInstances, inputInstances)
                this.centerFeatures = fe.extractFeatures(centerInstances);
                this.inputFeatures = fe.extractFeatures(inputInstances);

            elseif nargin <= 6
                %KLaplaceFC(widths, fe, centerInstances, inputInstances, ...
                %centerFeatures, inputFeatures)

                centerFeatures = in{5};
                inputFeatures = in{6};
                this.centerFeatures = centerFeatures;
                this.inputFeatures = inputFeatures;

            end

            this.options = this.getDefaultOptions();
        end

        % implementing abstract method
        function Kmat = kernelEval(this, F1, F2)
            lwidths = this.widths;
            F1sc = diag(1./lwidths)*F1;
            F2sc = diag(1./lwidths)*F2;
            D = sqrt(bsxfun(@plus, sum(F1sc.^2, 1)', sum(F2sc.^2, 1)) - 2*F1sc'*F2sc);
            Kmat = exp(-D);
        end



        function obj = finalize(this)
            % construct a dummy obj. Modify later.
            obj = KLaplaceFC(this.widths, this.featureExtractor, ...
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

