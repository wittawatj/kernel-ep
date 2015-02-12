classdef UAwareInstancesMapper < InstancesMapper
    %UAWAREINSTANCESMAPPER A map which takes an Instances and transforms it.
    %   - Uncertainty aware e.g., able to output a number quantifying its
    %   prediction uncertainty
    
    properties
    end
    
    methods (Abstract)

        % u: 1xn vector of uncertainties where n = length(instances). 
        U = estimateUncertainty(this, instances);
        
    end

    methods
        function [Zout, U] = mapInstancesAndU(this, Xin)
            % output both predictions and uncertainty
            % Subclass should override this if can be done more efficiently
            Zout = this.mapInstances(Xin);
            U = this.estimateUncertainty(Xin);
        end
    end
    
end

