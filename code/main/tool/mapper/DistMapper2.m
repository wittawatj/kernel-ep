classdef DistMapper2 < handle
    %DISTMAPPER2 A map taking two distributions as input and outputs
    %another distribution
    %   This is useful for representing an operation of gathering all
    %   incoming messages (two in this case) and output a projected message
    %   in EP.
    %
    %   Superseded by DistMapper. This class is deprecated.
    
    properties
    end
    
    methods (Abstract)
        % Produce an output distribution dout from two input distributions:
        % din1 and din2.
        dout = mapDist2(this, din1, din2);
        % Return a short summary string for this mapper
        s = shortSummary(this);
    end
    
end

