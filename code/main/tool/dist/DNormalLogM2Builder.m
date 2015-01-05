classdef DNormalLogM2Builder < DistNormalBuilder
    %DNORMALLOGM2BUILDER DistBuilder for DistNormal by construct to/from a normal with 
    %log uncentered 2nd moment
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
            M2 = [D.variance] + M1.^2;
            LogM2 = log(M2);
            S = [M1; LogM2];
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
            V = exp(S(2,:)) - M.^2;
            D = DistNormal(M, V);
        end

        function Scell = transformStat(this, X)
           Scell = cell(1, 2);
           [d,n] = size(X);
           if d==1
               Scell{1} = X;
               % log uncentred second moment not variance
               Scell{2} = log(X.^2);
           else
               % multivariate
               error('do not know how to take log matrix'); 
           end
        end

        function s = shortSummary(this)
            s = mfilename;
        end

        %%%%%%%%%%%%%%%%%%%%5
        
    end
    
    methods (Static)
    end
end

