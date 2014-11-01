classdef MVParamExtractor < FeatureExtractor
    %MVPARAMEXTRACTOR FeatureExtractor to extract mean and variance 
    %   .
    
    properties
    end
    
    methods
        % dist is a Distribution
        function F = extractFeatures(this, dist)
            assert(isa(dist, 'Distribution'));
            dist = DistArray(dist);
            M = [dist.mean];
            V = [dist.variance];
            if size(V, 1) > 1
                assert(unique(dist.d)>1);
                % multivariate Gaussian 
                V = reshape(V, [size(V, 1), size(V, 2), length(dist)]);
            end
            F = [M; V];
        end


        % Short summary of this FeatureMap. Useful if in the form
        % mapName(param1, param2).
        function s = shortSummary(this)
            s = sprintf('%s', mfilename);
        end
    end
    
end

