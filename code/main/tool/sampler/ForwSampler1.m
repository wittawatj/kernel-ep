classdef ForwSampler1 < handle
    %FORWSAMPLER1 A sampler for a conditional distribution of the form
    %p(x|y) where y is one variable. Hence the name.
%     
    
    properties
    end
    
    methods (Abstract)
        % Return samples X=[x_1, x_2, ...] where each x_i is drawn from
        % p(x_i | y_i)
        X = sampling1(this, Y);
      
    end
    
end

