classdef DistBetaBuilder < DistBuilder
    %DISTBETABUILDER DistBuilder for DistBeta
    
    properties
    end
    
    methods
        function S=getStat(this, D)
            assert(isa(D, 'DistBeta') || isa(D, 'DistArray'));
            assert(~isa(D, 'DistArray') || isa(D.distArray, 'DistBeta'));
            % stat is the first two moments.
            M = [D.mean];
            M2 = [D.variance] + M.^2;
            S = [M; M2];
        end
        
        function D=fromStat(this, S)
            assert(size(S,1)==2, 'Expected two rows for 1st and 2nd moments.');
            M = S(1,:);
            V = S(2,:)-M.^2;
            %beta parameters by method of moments
            % See https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters_2
            F = M.*(1-M)./V;
            A = M.*(F-1);
            B = (1-M).*(F-1);
            D = DistBeta(A, B);
        end
        
        function D= fromSamples(this, samples, weights)
            assert(size(samples, 1)==1, 'Beta samples must be 1d');
            assert(all(samples>=0 & samples <=1), 'Beta samples must be in [0,1].');
            
            % empirical mean
            m = samples*weights(:);
            % empirical 2nd moment
            m2 = (samples.^2)*weights(:);
            S = [m; m2];
            D = this.fromStat( S );
            
        end
        
        
        function L=empty(this, r, c)
            L = DistBeta.empty(r, c);
        end
        
        function s = shortSummary(this)
            s = mfilename;
        end
        
    end
    
end

