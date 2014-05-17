classdef Sampler < handle
    %SAMPLER Represent a sampler which can give some samples.
    
    properties
    end
    
    methods (Abstract)
        % Return samples X=[x_1, x_2, ...] 
        X = sampling0(this, N);
        
    end
    
end

