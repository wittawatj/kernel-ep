classdef DistMapper < handle & PrimitiveSerializable
    %DISTMAPPER A map taking incoming messages (Distribution) as input and outputs
    %another distribution
    %   This is useful for representing an operation of gathering all
    %   incoming messages (two in this case) and output a projected message
    %   in EP.
    
    properties
    end
    
    methods (Abstract)
        % Produce an output distribution dout from input distributions 
        % (instances of Distribution). varargin is a cell array of Distribution's
        dout=mapDists(this, varargin);

        % Produce a list of output Distribution's corresponding to a list of 
        % input Distribution's. varargin is a cell array of DistArray. 
        % Each DistArray corresponds to messages of a variable. 
        % All DistArray's must have the same length.
        % Return a DistArray
        douts=mapDistArrays(this, varargin);

        % Same as mapDists() with input messages represented by an instance 
        % of InMsgs
        dout=mapInMsgs(this, inMsgs);
        
        % return the number of incoming messages
        % this mapper expects to take.
        nv=numInVars(this);

        % Return a short summary string for this mapper
        s = shortSummary(this);
    end

    methods
        % convenient method to map all incoming messages in MsgBundle.
        % Return a DistArray of output messages.
        function outDa=mapMsgBundle(this, msgBundle)
            assert(isa(msgBundle, 'MsgBundle'));
            % cell array of DistArray's
            inDas=msgBundle.getInputBundles();
            outDa=this.mapDistArrays(inDas{:});
            
        end

    end
    
end

