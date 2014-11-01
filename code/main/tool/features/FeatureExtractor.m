classdef FeatureExtractor < handle
    %FEATUREEXTRACTOR Extract a finite dimensional feature vector from a distribution.
    %   .
    
    properties
    end
    
    methods(Abstract)
        % dist is a Distribution
        F = extractFeatures(this, dist);

        % Short summary of this FeatureMap. Useful if in the form
        % mapName(param1, param2).
        s = shortSummary(this);

        % Return the number of features to be generated.
        %D=getNumFeatures(this);
    end
    
end

