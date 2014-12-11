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
        
        function Mcell=getMoments(this, D)
            assert(isa(D, 'DistBeta'));
            n=length(D);
            Mcell=cell(1,n);
            for i=1:n
                m1=D(i).mean;
                assert(isscalar(m1));
                m2=D(i).variance+m1^2;
                Mcell{i}={m1, m2};
            end
        end

        function D=fromMoments(this, Mcell)
            assert(iscell(Mcell));
            n=length(Mcell);
            D=DistBeta.empty(0, n);
            for i=1:n
                m1=Mcell{i}{1};
                m2=Mcell{i}{2};
                S=[m1; m2];
                D(i)=this.fromStat(S);
            end
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
        
        function Scell = transformStat(this, X)
           M2 = X.^2;
           Scell = cell(1, 2);
           Scell{1} = X;
           Scell{2} = M2;
        end
        
        function L=empty(this, r, c)
            L = DistBeta.empty(r, c);
        end
        
        function s = shortSummary(this)
            s = mfilename;
        end
        
        % From PrimitiveSerializable interface 
        function s=toStruct(this)
            s.className=class(this);
        end
    end % end methods
    
end

