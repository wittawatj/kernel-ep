classdef GKConvolvable < handle
    %GKCONVOLVABLE Abstract distribution (message) which defines a method
    %for convolving with a Gaussian kernel.
    
    properties
    end
    
    methods (Abstract)
        % gw is a scalar for Gaussian kernel width
        
        Mux = conv_gaussian(this, X, gw)
    end
    
end

