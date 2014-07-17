classdef HasKLDivergence < handle
    %HASKLDIVERGENCE An interface marking a Distribution for which the
    %KL divergence can be computed.
    %    .
    
    properties
    end
    
    methods (Abstract)
        % Compute Hellinger distance from this distribution to the
        % distribution d2.
        div=klDivergence(this, d2);
            
    end
    
end

