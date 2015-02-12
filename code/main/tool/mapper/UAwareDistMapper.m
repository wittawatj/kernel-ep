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

        % Produce a list of uncertainty estimates for 
        % input Distribution's. varargin is a cell array of DistArray. 
        % Each DistArray corresponds to messages of a variable. 
        % All DistArray's must have the same length.
        % Return a matrix (or row vector) U.
        U = estimateUDistArrays(this, varargin)
    end

    methods 
        % A convenient method to get uncertainty of all message in MsgBundle
        function U = estimateUncertaintyMsgBundle(this, msgBundle)
            assert(isa(msgBundle, 'MsgBundle'));
            %nvars = length(msgBundle.inDistArrays);
            inDas = msgBundle.getInputBundles();
            U =this.estimateUDistArrays(inDas{:});

        end

    end
    
end

