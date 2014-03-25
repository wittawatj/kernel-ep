classdef Density < handle
    %DENSITY Density
    
    properties
    end
    
    methods (Abstract)
        % X (dxn)
        D=density(this, X)
    end
    
end

