classdef Distribution < handle
    %DIST A semantic tag for distributions.
    
    properties (Abstract, SetAccess=private)
        mean;
        variance;
    end
    
    methods (Abstract)
        
    end
    
end

