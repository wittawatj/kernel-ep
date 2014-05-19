classdef DistMapper2 < handle
    %DISTMAPPER2 A map taking two distributions as input and outputs
    %another distribution
    %   This is useful for representing an operation of gathering all
    %   incoming messages (two in this case) and output a projected message
    %   in EP.
    
    properties
    end
    
    methods (Abstract)
        % Produce an output distribution dout from two input distributions:
        % din1 and din2.
        dout = mapDist2(din1, din2);
    end
    
end

