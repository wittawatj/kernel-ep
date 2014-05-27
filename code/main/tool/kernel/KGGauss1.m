classdef KGGauss1 < Kernel
    %KGGAUSS1 Much like KGGaussian but each data variable is represented
    %with a struct instead of object array to avoid Matlab overhead.
    %  data variable is a struct with fields
    %  - mean = a matrix with each column representing one mean. dxn
    %  - variance = a row vector.
    %  - d = dimension of the distribution (= size(mean,1))
    %
    % Intended to be used with MV1Instances.
    %
    properties (SetAccess=private)
        % Gaussian width^2
        embed_width2
        width2;
        % Instance of KEGauss1
        keg1;
    end
    
    methods
        
        function this=KGGauss1(embed_width2, width2)
            % embed_width2  = Width for embedding into Gaussian RKHS
            % width2 = Gaussian width^2. Not the one for embedding the
            % distribution.
            assert(width2 > 0, 'Gaussian width must be > 0');
            assert(embed_width2 >0);
            this.embed_width2 = embed_width2;
            this.width2 = width2;
            this.keg1 = KEGauss1(embed_width2);
            
        end
        
        function Kmat = eval(this, s1, s2)
            %             assert(isstruct(s1));
            %             assert(isstruct(s2));
           
            w2 = this.width2;
            D2 = this.keg1.pairwise_dist2(s1, s2);
            Kmat = exp( -D2/(2*w2) );
            
        end
      
        function Kvec = pairEval(this, s1, s2)
            assert(isstruct(s1));
            assert(isstruct(s2));
            
            sig2 = this.embed_width2;
            w2 = this.width2;
            
            M1=s1.mean;
            M2=s2.mean;
            V1=s1.variance;
            V2=s2.variance;
            
            T1 = KEGauss1.self_inner1d(M1, V1, sig2);
            T2 = KEGauss1.self_inner1d(M2, V2, sig2);
            
            % width
            W = sig2 + V1+V2;
            D2 = (M1-M2).^2;
            E = exp(-D2./(2*W));
            % normalizer
            Z = sqrt(sig2./W);
            % ### hack to prevent negative W in case V1, V2 contain negative variances
            if any(imag(Z)>0)
                warning('In %s, kernel matrix contains imaginary entries.', mfilename);
            end
            Z(imag(Z)~=0) = 0;
            Cross = Z.*E;
            
            Kvec = exp(-(T1-2*Cross+T2)/(2*w2) );
            
        end
        
        function Param = getParam(this)
            Param = {this.embed_width2, this.width2};
        end
        
        function s=shortSummary(this)
            s = sprintf('KGGauss1(%.2g, %.2g)', this.embed_width2, this.width2 );
        end
    end
    
    
    methods (Static)
  
        function [DD, Med]=compute_meddistances(s, xembed_widths, subsamples)
            % for every embed width, compute the pairwise median distance
            % xembed_widths is a list.
            % High storage cost for DD.
            % - subsamples is a integer denoteing the size of subsamples
            % that will be used instead to compute the meddistance.
            assert(isstruct(s));
            n = size(s.mean, 2);
            if nargin >=3 && subsamples < n
                I = randperm(n, subsamples);
                s.mean = s.mean(I);
                s.variance = s.variance(I);
            end
            
            Med = zeros(1, length(xembed_widths));
            DD = cell(1, length(xembed_widths));
            for i=1:length(xembed_widths)
                sig2 = xembed_widths(i);
                kg = KEGauss1(sig2);
                D2 = kg.pairwise_dist2(s, s);
                DD{i} = D2;
                Med(i) = median(D2(:));
            end
            
        end
        
        function Kcell = candidates(s, embed_widths, med_factors, subsamples)
            % Generate a cell array of KGGauss1 candidates from a list of
            % embeding widths, embed_widths, and a list of factors med_factors
            % to be  multipled with the pairwise median distance of the mean
            % embeddings.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            
            if nargin < 4
                subsamples = length(s.mean);
            end
            assert(isstruct(s));
            assert(isnumeric(embed_widths));
            assert(~isempty(embed_widths));
            assert(isnumeric(med_factors));
            assert(~isempty(med_factors));
            assert(all(embed_widths > 0));
            assert(all(med_factors > 0));
            
            Ks = cell(length(embed_widths), length(med_factors));
            for i=1:length(embed_widths)
                ewidth = embed_widths(i);
                [~, med]= KGGauss1.compute_meddistances(s, ewidth, subsamples);
                for j=1:length(med_factors)
                    fac = med_factors(j);
                    w2 = fac*med;
                    Ks{i,j} = KGGauss1(ewidth, w2);
                end
            end
            Kcell = reshape(Ks, [1, length(embed_widths)*length(med_factors)]);
        end
        
    end
end

