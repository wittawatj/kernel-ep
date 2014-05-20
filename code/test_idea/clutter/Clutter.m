classdef Clutter
    %CLUTTER All helpers related to clutter problem
    
    
    properties
    end
    
    methods (Static)
        
        function [Theta, tdist] = theta_dist(N)
            % Theta distribution on testing
            var_theta = 0.1;
            mu = 3;
            dis_theta = DistNormal(mu, var_theta);
            Theta = dis_theta.draw(N);
            tdist = @(t)(normpdf(t, mu, sqrt(var_theta) ));
        end
        
        
        
        
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
        
    end %end static methods
    
end

