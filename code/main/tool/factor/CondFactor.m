classdef CondFactor < handle
    %CONDFACTOR A class representing a conditional factor.
    %   . 
    
    properties (SetAccess = protected)
    end
    
    methods (Abstract)

        % return the number of incoming variables (connected variables to a factor)
        d=numInVars(this);

        % sample from this factor conditioning on cond_samples. 
        % cond_samples must be an instance of MatTensorInstances.
        points = sample(this, cond_samples);

        % Draw a sequence of K quasi-monte carlo conditioning points 
        % (low discrepancy sequence). The resuling MatTensorInstances can be fed 
        % into this.sample(). Union of multiple draws does not necessarily yield 
        % the same sample set as one big batch draw. 
        %
        matTenIns = batchDrawQuasiMCPoints(this, K);

        % return a short summary of this factor
        s = shortSummary(this);
    end
    
end

