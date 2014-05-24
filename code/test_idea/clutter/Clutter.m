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
        
    end %end static methods
    
end

