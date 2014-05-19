classdef DistMapper2 < handle
    %DISTMAPPER2 A map taking two distributions as input and outputs
    %another distribution
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Abstract)
        % Produce an output distribution dout from two input distributions:
        % din1 and din2.
        dout = mapDist2(din1, din2);
    end
    
end

