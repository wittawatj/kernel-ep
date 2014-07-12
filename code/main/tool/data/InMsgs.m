classdef InMsgs < handle
    %INMSGS A class representing on tensor instance of incoming messages.
    
    properties(SetAccess=protected)
        % a cell array of Distribution's
        msgs;
    end
    
    methods
        function this=InMsgs(varargin)
            assert(~isempty(varargin));
            C = varargin;
            for i=1:length(C)
                assert(isa(C{i}, 'Distribution'));
            end
            this.msgs = C;
        end

        function v=numInVars(this)
            % Return the number of variables
            v=length(this.msgs);
        end

        function dist=getMsg(this, index)
            % Return the message at the index.
            % index in [1, numInVars()]
            assert(index>=1 && index<=this.numInVars());
            dist=this.msgs{index};

        end

        function msgs=getMsgs(this)
            msgs=this.msgs;
        end


    end
    
end

