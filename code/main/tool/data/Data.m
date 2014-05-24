classdef Data
    %DATA Helper functions related to data
    %   
    
    properties
    end
    
    methods (Static)
        
        function [X, T, Tout, Xte, Tte, Toutte] = splitTrainTest(s, ntr, nte)
            %  s is likely to be loaded from  l_clutterTrainMsgs.
            eval(structvars(100, s));
            
            % respect ntr first
            toremove = length(X)-min(ntr, length(X));
            Id = randperm( length(X),  toremove);
            Xte = X(Id);
            X(Id) = [];
            
            Tte = T(Id);
            T(Id) = [];
            
            Toutte = Tout(Id);
            Tout(Id) = [];
            
            % further reduce to nte if nte < n-ntr
            nte = min(length(Xte), nte);
            Id2 = randperm(length(Xte), nte);
            Xte = Xte(Id2);
            Tte = Tte(Id2);
            Toutte = Toutte(Id2);
        end
        
    end
    
end

