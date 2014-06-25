classdef InstancesMapper < handle
    %INSTANCESMAPPER A map which takes an Instances and transforms it.
    %   There is no restrction on the output of the map.
    
    properties
    end
    
    methods (Abstract)
        % Map Instances Xin into Zout. The type of Zout is not restricted.
        Zout = mapInstances(this, Xin);

        % return a short summary of this mapper
        s = shortSummary(this);

    end
    
end

