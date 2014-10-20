classdef DNormalVarBuilder < DistNormalBuilder
    %DNORMALVARBUILDER DistBuilder for DistNormal by construct to/from a normal with 
    % variance
    %    Work only for 1d Gaussians.
    
    properties
    end
    
    methods
      
        function L=empty(this, r, c)
            L = DistNormal.empty(r, c);
        end
        
        function S=getStat(this, D)
            assert(isa(D, 'DistNormal') || isa(D, 'DistArray'));
            assert(~isa(D, 'DistArray') || isa(D.distArray, 'DistNormal'));
            dim = unique(D.d);
            assert(length(dim)==1);
            assert(dim==1);

            M1 = [D.mean];
            Variance = [D.variance];
            S = [M1; Variance];
        end
        
        function D=fromStat(this, S)
            % Construct DistNormal(s) from a sufficient statistics returned
            % from getStat()
            l = size(S, 1);
            dim = DistNormalBuilder.findDim(l);
            if isempty(dim)
                error('Invalid dimension of stat.');
            end
            assert(length(dim)==1);
            assert(dim==1);
            M = S(1,:);
            V = S(2,:);
            D = DistNormal(M, V);
        end

        function s = shortSummary(this)
            s = mfilename;
        end

        %%%%%%%%%%%%%%%%%%%%5
        
    end
    
    methods (Static)
    end
end

