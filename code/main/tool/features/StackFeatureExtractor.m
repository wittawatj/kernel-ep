classdef StackFeatureExtractor < FeatureExtractor
    %STACKFEATUREEXTRACTOR A FeatureExtractor which stack multiple FeatureExtractor's. 
    %   - For input of type TensorInstances.
    %   - The number of inputs in TensorInstances must be the same as the number of 
    %   FeatureExtractor's.
    %
    
    properties(SetAccess = protected)
        % cell array of FeatureExtractor's
        featureExtractors;
    end
    
    methods
        function this = StackFeatureExtractor(varargin)
            fExtractors = varargin;
            assert(iscell(fExtractors));
            assert(all(cellfun(@(f)isa(f, 'FeatureExtractor'), fExtractors )));
            this.featureExtractors = fExtractors;
        end

        % dist is a Distribution
        function F = extractFeatures(this, tensorIns)
            assert(tensorIns.tensorDim() == length(this.featureExtractors));
            numInVars = tensorIns.tensorDim();
            Fs = cell(1, numInVars);
            for i=1:numInVars
                fe = this.featureExtractors{i};
                oneIns = tensorIns.instancesCell{i};
                Fs{i} = fe.extractFeatures(oneIns);
            end
            F = vertcat(Fs{:});
        end


        % Short summary of this FeatureMap. Useful if in the form
        % mapName(param1, param2).
        function s = shortSummary(this)
            s = sprintf('%s', mfilename);
        end
    end
    
end

