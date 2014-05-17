classdef KEGaussian < Kernel
    %KEGAUSSIAN Kernel for distributions defined as the inner product of
    %their mean embeddings into Gaussian RKHS.
    %
    % Expected product Gaussian kernel for mean embeddings of Gaussian
    % distributions. Equivalently, compute the inner product of mean embedding
    % using Gaussian kernel.
    %
    % Return K which is a n1 x n2 matrix where K_ij represents the inner
    % product of the mean embeddings of Gaussians in the RKHS induced by the
    % Gaussian kernel with width sigma2. Since the mapped distributions and the
    % kernel are Gaussian, the kernel evaluation can be computed analytically.
    %
    % At present, only work for array of DistNormal's (as data).
    %
    
    properties (SetAccess=private)
        % Gaussian width^2
        sigma2;
    end
    
    methods
        function this=KEGaussian(sigma2)
            % sigma2 = Gaussian width^2 used for embedding into Gaussian
            % RKHS
            assert(sigma2 > 0, 'Gaussian width must be > 0');
            this.sigma2 = sigma2;
        end
        
        
        function Kmat = eval(this, data1, data2)
            Kmat = kerEGaussian(data1, data2, this.sigma2);
        end
        
        
        function Kvec = pairEval(this, X, Y)
            assert(isa(X, 'DistNormal'));
            assert(isa(Y, 'DistNormal'));
            assert(length(X)==length(Y));
            n = length(X);
            sig2 = this.sigma2;
            if X(1).d==1
                kg1 = KEGauss1(this.sigma2);
                ins1= Gauss1Instances(X);
                ins2 = Gauss1Instances(Y);
                s1=ins1.getAll();
                s2=ins2.getAll();
                Kvec = kg1.pairEval(s1, s2);
            else
                error('later for multivariate case.');
            end
        end
        
        function Param = getParam(this)
            Param = {this.sigma2};
        end
        
        function s=shortSummary(this)
            s = sprintf('KEGaussian(%.3g)', this.sigma2 );
        end
    end
    
end

