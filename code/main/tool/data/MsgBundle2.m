classdef MsgBundle2 < handle
    %MSGBUNDLE2 A container for mapper learner of factor p(x|t) 
    % taking 2 incoming messages.
    % The container contains left messages x, right messages t and target
    % output messages.
    %
    % Deprecated. This class is superseded by MsgBundle.
    %
    %
    
    properties
    end
    
    methods (Abstract)
        % Return an array of Distribution's representing messages from x in
        % p(x|t)
        msgs = getLeftMessages(this);
        
        % Return an array of Distribution's representing messages from t in
        % p(x|t)
        msgs = getRightMessages(this);
        
        % Return an array of ground truth output messages.
        msgs = getOutMessages(this);
    
        n = count(this);
        
    end
    
end

