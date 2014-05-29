classdef HasHellingerDistance < handle
    %HASHELLINGERDISTANCE An interface marking a distribution for which the
    %Hellinger distance can be computed.
    %    See https://en.wikipedia.org/wiki/Hellinger_distance
    
    properties
    end
    
    methods (Abstract)
        % Compute Hellinger distance from this distribution to the
        % distribution d2.
        dist=distHellinger(this, d2);
            
    end
    
end

