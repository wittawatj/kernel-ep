classdef FactorOperator < handle & PrimitiveSerializable
    %FACTOROPERATOR A container of all DistMapper's to all directions. 
    %   This object is what sits in a factor and handles all incoming messages 
    %   and produces outgoing messages. 
    %
    
    properties
    end
    
    methods (Abstract)
        % Return a DistMapper given the variable index ranging from 1 to 
        % numInVars(). Assume the factor takes the form p(x_1 | x_2, ..).
        % So index=1 always refers to x_1. 
        dm=getDistMapper(this, index);

        % return the number of incoming messages
        % this mapper expects to take.
        nv=numInVars(this);

        % Return a short summary string for this mapper
        s=shortSummary(this);
    end
    
end

