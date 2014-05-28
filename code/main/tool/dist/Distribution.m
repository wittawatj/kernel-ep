classdef Distribution < handle
    %DIST A semantic tag for distributions.
    
    properties (Abstract, SetAccess=protected)
        mean;
        variance;
        
        % A cell array of parameters in the usual parametrization. For
        % Gaussian for example, this should be {mean, variance}.
        parameters;
    end
    
    methods (Abstract)
        
    end
    
end

