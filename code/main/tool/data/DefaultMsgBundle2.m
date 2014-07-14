classdef DefaultMsgBundle2 < MsgBundle2
    %DEFAULTMSGBUNDLE2 Default implementation of MsgBundle2
    %
    % This class is deprecated. Consider using DefaultMsgBundle.
    %
    
    properties (SetAccess=protected)
        % X messages in p(x|t).
        leftMessages;
        rightMessages;
        outMessages;
    end
    
    methods
        function this=DefaultMsgBundle2(leftMsgs, rightMsgs, out)
            assert(isa(leftMsgs, 'Distribution'));
            assert(isa(rightMsgs, 'Distribution'));
            assert(isa(out, 'Distribution'));
            assert(length(leftMsgs)==length(rightMsgs));
            assert(length(rightMsgs)==length(out));
            
            this.leftMessages = leftMsgs;
            this.rightMessages = rightMsgs;
            this.outMessages = out;
        end
        
        function msgs = getLeftMessages(this)
            msgs = this.leftMessages;
        end
        
        function  msgs = getRightMessages(this)
            msgs = this.rightMessages;
        end
        
        function msgs = getOutMessages(this)
            msgs = this.outMessages;
        end
        
        function n = count(this)
            n = length(this.outMessages);
        end
        
        function [newMsgBundle] = removeRandom(this, nremove)
            % Randomly remove some messages and return a new MsgBundle.
            % nremove cannot be more than count().
            %
            assert(nremove < this.count(), 'nremove must be < count()' );
            total = this.count();
            Id = randperm( total,  nremove);
            
            leftR = this.leftMessages(Id);
            rightR = this.rightMessages(Id);
            outR = this.outMessages(Id);
            
            this.leftMessages(Id) = [];
            this.rightMessages(Id) = [];
            this.outMessages(Id) = [];
            
            newMsgBundle = DefaultMsgBundle2(leftR, rightR, outR);
            assert(newMsgBundle.count()+this.count() == total);
        end
        
        function reduceTo(this, nto)
            % Reduce the bundle by removing some messages so that count()<=nto
            if nto >= this.count()
                return;
            end
            total = this.count();
            Id = randperm(total,  total-nto);
            
            this.leftMessages(Id) = [];
            this.rightMessages(Id) = [];
            this.outMessages(Id) = [];
            
            assert(this.count() == nto);
        end
        
        %%%%%%%%%%%%%
        function s = saveobj(this)
            s.LM = this.leftMessages;
            s.RM = this.rightMessages;
            s.OM = this.outMessages;
        end
    
    end
    
    methods (Static)
        function obj = loadobj(s)
            
            % loadobj must be Static so it can be called without object
            obj = DefaultMsgBundle2(s.LM, s.RM, s.OM);
        end
    end
    
end

