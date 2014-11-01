classdef NatParamExtractor < FeatureExtractor
    %NATPARAMEXTRACTOR Extract natural parameters.  
    %   - Extract log precision for Gaussian instead of just precisions.
    
    properties
    end
    
    methods
        % dist is a Distribution
        function F = extractFeatures(this, dist)
            % dist can be a DistArray
            dt = dist.getDistType();
            if strcmp(dt, 'DistNormal') && unique([dist.d])==1
                M = [dist.mean];
                V = [dist.variance];
                nat1 = M./V;
                logPre = -log(V);
                F = [nat1; logPre];
            elseif isa(dist, 'DistArray') && strcmp(dt, 'DistBeta')
                inDa = dist.distArray;
                a = [inDa.alpha]-1;
                b = [inDa.beta]-1;
                F = [a; b];
            elseif isa(dist, 'DistBeta')
                a = [dist.alpha]-1;
                b = [dist.beta]-1;
                F = [a; b];
            else 
                error('Unknown supported distribution type %s: ', class(dist(1)));
            end


        end


        % Short summary of this FeatureMap. Useful if in the form
        % mapName(param1, param2).
        function s = shortSummary(this)
            s = sprintf('%s', mfilename);
        end
    end
    
end

