classdef KParams2Gauss1 < Kernel
    %KPARAMS2GAUSS1 Product of two Gaussian kernels, one on the first
    %parameter, the other on the second. Assume
    %
    %  data variable is a struct with fields
    %  - param1 = a matrix with each column representing 1st parameter d1xn
    %  - param2 = a matrix with each column representing 2nd parameter d2xn
    %
    % Intended to be used with Params2Instances
    %
    
    properties (SetAccess=private)
        % Gaussian width^2 on param1
        param1_width2
        % Gaussian width^2 on param2
        param2_width2;
        
    end
    
    methods
        
        function this=KParams2Gauss1(p1width2, p2width2)
            
            assert(p1width2 > 0, 'Gaussian width on param1 must be > 0.');
            assert(p2width2 >0, 'Gaussian width on param2 must be > 0.');
            this.param1_width2 = p1width2;
            this.param2_width2 = p2width2;
            
            warning(['usage of %s is discouraged. '...
                'Should try to unify it with DistArray.'], mfilename);
        end
        
        function Kmat = eval(this, s1, s2)
            %             assert(isstruct(s1));
            %             assert(isstruct(s2));
            M1=s1.param1;
            M2=s2.param1;
            V1=s1.param2;
            V2=s2.param2;
            
            % pairwise distance^2 on params
            DM2 = bsxfun(@minus, M1', M2).^2;
            DV2 = bsxfun(@minus, V1', V2).^2;
            
            mw2 = this.param1_width2;
            vw2 = this.param2_width2;
            
            Kmat = exp( -DM2/(2*mw2) -DV2/(2*vw2) );
        end
        
        function Kvec = pairEval(this, s1, s2)
            assert(isstruct(s1));
            assert(isstruct(s2));
            M1=s1.param1;
            M2=s2.param1;
            V1=s1.param2;
            V2=s2.param2;
            
            DM2 = (M1-M2).^2;
            DV2 = (V1-V2).^2;
            
            mw2 = this.param1_width2;
            vw2 = this.param2_width2;
            
            Kvec = exp( -DM2/(2*mw2) -DV2/(2*vw2) );
        end
        
        function Param = getParam(this)
            Param = {this.param1_width2, this.param2_width2};
        end
        
        function s=shortSummary(this)
            s = sprintf('KParams2Gauss1(%.2g, %.2g)', this.param1_width2, ...
                this.param2_width2 );
        end
    end
    
    
    methods (Static)
        function Kcell = candidates(s, param1_medf, param2_medf, subsamples)
            % - Generate a cell array of kernel candidates from a list of
            % param1_medf, a list of factors to be  multipled with the
            % pairwise median distance of param1
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            assert(isstruct(s));
            n = length(s.param1);
            if nargin < 4
                subsamples = n;
            end
            assert(isnumeric(param1_medf));
            assert(~isempty(param1_medf));
            assert(isnumeric(param2_medf));
            assert(~isempty(param2_medf));
            assert(all(param1_medf > 0));
            assert(all(param2_medf > 0));
            % subsampling if needed
            if subsamples < n
                I = randperm(n, subsamples);
                s.param1= s.param1(I);
                s.param2 = s.param2(I);
            end
            
            Ks = cell(length(param1_medf), length(param2_medf));
            % !! don't forget the ^2 here !
            med1 = meddistance(s.param1)^2;
            med2 = meddistance(s.param2)^2;
            for i=1:length(param1_medf)
                medf1 = param1_medf(i);
                for j=1:length(param2_medf)
                    medf2 = param2_medf(j);
                    
                    param1_width2 = med1*medf1;
                    param2_width2 = med2*medf2;
                    
                    Ks{i,j} = KParams2Gauss1(param1_width2, param2_width2);
                end
            end
            Kcell = reshape(Ks, [1, length(param1_medf)*length(param2_medf)]);
        end
        
    end
end

