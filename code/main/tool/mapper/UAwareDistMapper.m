classdef UAwareDistMapper < DistMapper
    %UAWAREDISTMAPPER A map taking incoming messages (Distribution) as input and 
    %outputs another distribution. The map is aware of its own prediction uncertainty.
    %
    
    properties
    end
    
    methods (Abstract)
        % Produce an uncertainty estimate (a real number) of the given inputs. 
        % varargin is a cell array of Distribution's. This is used in conjunction 
        % with mapDist(..). Lower u is better. For example, u might be predictive
        % variance.
        u = estimateUncertainty(this, varargin);

        % output both predictions and uncertainty
        [douts, U] = mapDistsAndU(this, varargin);
    end
    
end

