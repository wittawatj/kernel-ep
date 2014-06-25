classdef KNaturalGauss1 < Kernel
    %KNATURALGAUSS1 Product of two Gaussian kernels on the two natural
    %parameters of a Gaussian: mu/var^2, -1/(2*var^2).
    %
    %  data variable is a struct with fields
    %  - mean = a matrix with each column representing one mean. dxn
    %  - variance = a row vector.
    %  data can be a DistArray 
    %
    % Intended to be used with MV1Instances. This kernel behaves much
    % like KMVGauss1.
    %
    properties (SetAccess=private)
        % Gaussian width^2 on mu/var^2
        precMeanWidth2;
        % Gaussian width^2 on -1/(2*var^2).
        negPrecWidth2;
         
    end
    
    methods
        
        function this=KNaturalGauss1(prec_mean_w, neg_prec_w)
            
            assert(prec_mean_w> 0, 'Gaussian width must be > 0.');
            assert(neg_prec_w >0, 'Gaussian width  must be > 0.');
            this.precMeanWidth2 = prec_mean_w;
            this.negPrecWidth2 = neg_prec_w;
            
        end
        
        function Kmat = eval(this, s1, s2)
            assert(isa(s1, 'Distribution') || isstruct(s1));
            % precision x mean, and negative precision
            [PM1, NP1]=KNaturalGauss1.toNaturalParams(s1.mean, s1.variance);
            [PM2, NP2]=KNaturalGauss1.toNaturalParams(s2.mean, s2.variance);
            
            % pairwise distance^2 
            dpm2 = bsxfun(@minus, PM1', PM2).^2;
            dnp2 = bsxfun(@minus, NP1', NP2).^2;
            
            pm_width2 = this.precMeanWidth2;
            np_width2 = this.negPrecWidth2;
            
            Kmat = exp( -dpm2/(2*pm_width2) -dnp2/(2*np_width2) );
        end
      
        function Kvec = pairEval(this, s1, s2)
            assert(isstruct(s1) || isa(s1, 'Distribution'));
            assert(isstruct(s2) || isa(s2, 'Distribution'));
            
            % precision x mean, and negative precision
            [PM1, NP1]=KNaturalGauss1.toNaturalParams(s1.mean, s1.variance);
            [PM2, NP2]=KNaturalGauss1.toNaturalParams(s2.mean, s2.variance);
            
            % pairwise distance^2 
            dpm2 = (PM1-PM2).^2;
            dnp2 = (NP1-NP2).^2;
            
            pm_width2 = this.precMeanWidth2;
            np_width2 = this.negPrecWidth2;
            
            Kvec = exp( -dpm2/(2*pm_width2) -dnp2/(2*np_width2) );
            
        end
        
        function Param = getParam(this)
            Param = {this.precMeanWidth2, this.negPrecWidth2};
        end
        
        function s=shortSummary(this)
            s = sprintf('KNaturalGauss1(%.2g, %.2g)', this.precMeanWidth2, ...
                this.negPrecWidth2 );
        end
    end
    
    
    methods (Static)
        function [PM, NP]=toNaturalParams(Mean, Var)
            % Change from Mean, Variance into natural parameters
            Vsq = Var.^2;
            PM=Mean./Vsq;
            % negative precision
            NP= -0.5./Vsq;
            
        end
        
        function Kcell = candidates(s, pm_medf, np_medf, subsamples)
            % - Generate a cell array of kernel candidates from a list of
            % pm_medf, a list of factors to be  multipled with the 
            % pairwise median distance of the precision*mean.
            % - Same semantic for np_medf but for the -0.5*precision.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            assert(isstruct(s));
            n = length(s.mean);
            if nargin < 4
                subsamples = n;
            end
            assert(isnumeric(pm_medf));
            assert(~isempty(pm_medf));
            assert(isnumeric(np_medf));
            assert(~isempty(np_medf));
            assert(all(pm_medf > 0));
            assert(all(np_medf > 0));
            % subsampling if needed
            if subsamples < n
                I = randperm(n, subsamples);
                s.mean = s.mean(I);
                s.variance = s.variance(I);
            end
            
            Ks = cell(length(pm_medf), length(np_medf));
            
            % precision x mean, and negative precision
            [PM, NP]=KNaturalGauss1.toNaturalParams(s.mean, s.variance);
            
            % !! don't forget the ^2 here !
            pm_med = meddistance(PM)^2;
            np_med = meddistance(NP)^2;
            for i=1:length(pm_medf)
                pmfac = pm_medf(i);
                for j=1:length(np_medf)
                    npfac = np_medf(j);
                    pm_width2 = pmfac*pm_med;
                    np_width2 = npfac*np_med;
                    
                    Ks{i,j} = KNaturalGauss1(pm_width2, np_width2);
                end
            end
            Kcell = reshape(Ks, [1, length(pm_medf)*length(np_medf)]);
        end
        
    end
end

