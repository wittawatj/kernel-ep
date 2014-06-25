classdef KMVGauss1 < Kernel
    %KMVGauss1 Product of two Gaussian kernels, one on means, the other on
    % variance.
    %
    %  data variable is a struct with fields
    %  - mean = a matrix with each column representing one mean. dxn
    %  - variance = a row vector.
    %  data variable can also be a DistArray.
    %
    % Intended to be used with MV1Instances.
    %
    properties (SetAccess=private)
        % Gaussian width^2 on means
        mean_width2
        % Gaussian width^2 on variances
        var_width2;
         
    end
    
    methods
        
        function this=KMVGauss1(mwidth2, vwidth2)
            
            assert(mwidth2 > 0, 'Gaussian width on means must be > 0.');
            assert(vwidth2 >0, 'Gaussian width on variances must be > 0.');
            this.mean_width2 = mwidth2;
            this.var_width2 = vwidth2;
            
        end
        
        function Kmat = eval(this, s1, s2)
            %             assert(isstruct(s1));
            %             assert(isstruct(s2));
            M1=s1.mean;
            M2=s2.mean;
            V1=s1.variance;
            V2=s2.variance;
            
            % pairwise distance^2 on means
            DM2 = bsxfun(@minus, M1', M2).^2;
            DV2 = bsxfun(@minus, V1', V2).^2;
            
            mw2 = this.mean_width2;
            vw2 = this.var_width2;
            
            Kmat = exp( -DM2/(2*mw2) -DV2/(2*vw2) );
        end
      
        function Kvec = pairEval(this, s1, s2)
            % s1, s2 can be a DistArray or Distribution. 
            % The only things KMVGauss1 needs are .mean and .variance which are 
            % in Distribution.
            assert(isstruct(s1) || isa(s1, 'Distribution'));
            assert(isstruct(s2) || isa(s2, 'Distribution'));
            M1=s1.mean;
            M2=s2.mean;
            V1=s1.variance;
            V2=s2.variance;
            
            DM2 = (M1-M2).^2;
            DV2 = (V1-V2).^2;
            
            mw2 = this.mean_width2;
            vw2 = this.var_width2;
            
            Kvec = exp( -DM2/(2*mw2) -DV2/(2*vw2) );
        end
        
        function Param = getParam(this)
            Param = {this.mean_width2, this.var_width2};
        end
        
        function s=shortSummary(this)
            s = sprintf('KMVGauss1(%.2g, %.2g)', this.mean_width2, ...
                this.var_width2 );
        end
    end
    
    
    methods (Static)
        
        function Kcell = candidates(s, mean_medf, var_medf, subsamples)
            % - Generate a cell array of kernel candidates from a list of
            % mean_medf, a list of factors to be  multipled with the 
            % pairwise median distance of the means.
            % - Same semantic for var_medf but for the variance.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            assert(isstruct(s));
            n = length(s.mean);
            if nargin < 4
                subsamples = n;
            end
            assert(isnumeric(mean_medf));
            assert(~isempty(mean_medf));
            assert(isnumeric(var_medf));
            assert(~isempty(var_medf));
            assert(all(mean_medf > 0));
            assert(all(var_medf > 0));
            % subsampling if needed
            if subsamples < n
                I = randperm(n, subsamples);
                s.mean = s.mean(I);
                s.variance = s.variance(I);
            end
            
            Ks = cell(length(mean_medf), length(var_medf));
            % !! don't forget the ^2 here !
            mean_med = meddistance(s.mean)^2;
            var_med = meddistance(s.variance)^2;
            for i=1:length(mean_medf)
                mmedf = mean_medf(i);
                for j=1:length(var_medf)
                    vmedf = var_medf(j);
                    
                    mean_width2 = mmedf*mean_med;
                    var_width2 = vmedf*var_med;
                    
                    Ks{i,j} = KMVGauss1(mean_width2, var_width2);
                end
            end
            Kcell = reshape(Ks, [1, length(mean_medf)*length(var_medf)]);
        end
        
    end
end

