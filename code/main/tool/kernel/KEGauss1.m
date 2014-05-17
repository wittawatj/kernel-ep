classdef KEGauss1 < Kernel
    %KEGAUSS1 Much like KEGaussian but each data variable is represented
    %with a struct instead of object array to avoid Matlab overhead.
    %  data variable is a struct with fields
    %  - mean = a matrix with each column representing one mean. dxn
    %  - variance = If d==1, a row vector. If d>1, a dxdxn matrix.
    %  - d = dimension of the distribution (= size(mean,1))
    %
    % Intended to be used with Gauss1Instances.
    %
    properties (SetAccess=private)
        % Gaussian width^2
        sigma2;
    end
    
    methods
        function this=KEGauss1(sigma2)
            % sigma2 = Gaussian width^2 used for embedding into Gaussian
            % RKHS
            assert(sigma2 > 0, 'Gaussian width must be > 0');
            this.sigma2 = sigma2;
        end
        
        
        function Kmat = eval(this, s1, s2)
            P1 = [s1.mean; s1.variance];
            P2 = [s2.mean; s2.variance];
            Kmat = kerEGaussian1(P1, P2, this.sigma2);
        end
        
          
        function D2=pairwise_dist2(this, s1, s2)
            % a matrix of pairwise distance^2 
            sig2 = this.sigma2;
            n1 = length(s1.mean);
            n2 = length(s2.mean);
            
            T1 = repmat(this.pairEval(s1, s1)', 1, n2);
            T2 = repmat(this.pairEval(s2, s2), n1, 1);
            
            P1 = [s1.mean; s1.variance];
            P2 = [s2.mean; s2.variance];
            Cross = kerEGaussian1(P1, P2, sig2);
            
            D2 = T1 -2*Cross + T2 ;
        end
        
        
        function Kvec = pairEval(this, s1, s2)
            
            n = length(s1.mean);
            sig2 = this.sigma2;
            
            % operation on obj array can be expensive..
            M1 = s1.mean;
            V1 = s1.variance;
            M2 = s2.mean;
            V2 = s2.variance;
            
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
            Kvec = Z.*E;
            
        end
        
        function Param = getParam(this)
            Param = {this.sigma2};
        end
        
        function s=shortSummary(this)
            s = sprintf('KEGaussian(%.3g)', this.sigma2 );
        end
    end
    
    methods (Static)
        
        function C=self_inner1d(M, V, sigma2)
            % return C as a row vector. Efficient implementation
            
            % width
            W = sigma2 + 2*V;
            % normalizer
            Z = sqrt(sigma2./W);
            % ### hack to prevent negative W in case V1, V2 contain negative variances
            if any(imag(Z)>0)
                warning('In %s, kernel matrix contains imaginary entries.', mfilename);
            end
            Z(imag(Z)~=0) = 0;
            % the exponential part is 1.
            C = Z;
            
        end
    end
    
end

