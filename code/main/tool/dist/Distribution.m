classdef (Abstract) Distribution < handle
    %DIST A semantic tag for distributions.
    
    properties
    end
    
    methods (Abstract)
        
        % return the first moment
        m=getMean(this);
        
        % return variance
        v=getVariance(this);
    end
    
end

