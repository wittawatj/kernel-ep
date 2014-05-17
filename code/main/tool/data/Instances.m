classdef Instances < handle
    %INSTANCES Abstract class representing data instances.
    
    properties
    end
    
    methods (Abstract)
        % Return data instances specified by the indices in Ind. Ind is a
        % list of indices. The type of Data is not restricted (depending on
        % the implmentation).
        Data=get(this, Ind);
        
        % Return all data instances.
        Data=getAll(this);
        
        % Return data instances specified by the indices in Ind in the form 
        % of Instances. Ind is a list of indices.
        Ins=instances(this, Ind);
        
        % total number of instances 
        l = count(this);
    end
    
end

