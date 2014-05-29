classdef DistBetaBuilder < DistBuilder
    %DISTBETABUILDER DistBuilder for DistBeta
    
    properties
    end
    
    methods
        function D= fromSamples(this, samples, weights)
            assert(size(samples, 1)==1, 'Beta samples must be 1d');
            assert(all(samples>=0 & samples <=1), 'Beta samples must be in [0,1].');
            
            % empirical mean
            m = samples*weights(:);
            % empirical variance
            v = (samples.^2)*weights(:) - m^2;
            
            %beta parameters by method of moments
            % See https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters_2
            f = m*(1-m)/v;
            a = m*(f-1);
            b = (1-m)*(f-1);
            D = DistBeta(a, b);
        end
        
        
        function L=empty(this, r, c)
            L = DistBeta.empty(r, c);
        end
        
        
    end
    
end

