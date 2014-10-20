classdef LinearFMInstancesMapper < InstancesMapper 
    %LINEARFMINSTANCESMAPPER InstancesMapper given by the use of a FeatureMap with  
    %a transformation matrix to generate outputs.
    %   .
    
    properties
        % Transformation matrix Dout x D where D is the #features 
        W;

        featureMap;
    end
    
    methods
        function this=LinearFMInstancesMapper(W, featureMap)
            assert(isnumeric(W));
            assert(isa(featureMap, 'FeatureMap'));
            this.W = W;
            this.featureMap = featureMap;
        end

        function Zout = mapInstances(this, Xin)
            % Map Instances in Xin to Zout with this operator.
            assert(isa(Xin, 'Instances'));
            fm = this.featureMap;
            Pin = fm.genFeatures(Xin);
            Zout = this.W*Pin;
        end

        function s = shortSummary(this)
            s = sprintf('%s(%s)', mfilename, this.featureMap.shortSummary());
        end

    end
    
end

