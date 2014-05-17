classdef CondOpGGauss2 < handle
    %CONDOPGGAUSS2 Conditional mean embedding operator taking 2 messages
    %(distributions) and producing one outgoing message. Use Gaussian
    % kernel on mean embeddings (kerGGaussian). 
    %   The outgoing message is a Gaussian computed by a finite-dimensional
    %   feature map of the form (z, z^2).
    % Conditional mean embedding operator: C_{z'|xz}
    
    properties (SetAccess=private)
        f1sample %X. f1 = from the first
        f2sample %Y
        tsample %Z
        gw1 % Parameter of Gaussian kernel on mean embeddings
        gw2 
        embed_width1 % parameter for mean embeddings of f1sample
        embed_width2 % parameter width for mean embeddings of f2sample
        operator;
        
        % first-order moments of tsample
        t1moments;
        t2moments;
    end
    
    methods
        % f = from
        % t = to
        % gw1, gw2 = Gaussian width of kernel associated with this operator
        % lamb = regularization parameter
        function this = CondOpGGauss2(f1sample, f2sample, tsample, ...
                operator, embed_width1, gw1, embed_width2, gw2)
            assert(size(f1sample, 2) == size(f2sample, 2));
            assert(size(f2sample, 2) == size(tsample, 2));
            assert(isa(f1sample, 'DistNormal'));
            assert(isa(f2sample, 'DistNormal'));
            assert(isa(tsample, 'DistNormal'));
            assert(gw1 > 0);
            assert(gw2 > 0);
            assert(embed_width1 > 0);
            assert(embed_width2 > 0);
            
            this.f1sample = f1sample;
            this.f2sample = f2sample;
            this.tsample = tsample;
            this.embed_width1 = embed_width1;
            this.gw1 = gw1;
            this.embed_width2 = embed_width2;
            this.gw2 = gw2;
            this.operator = operator;
            
            % cached moments
            td = tsample(1).d;
            if td > 1
                error('Multivariate Gaussians are not supported yet.');
            end
            TM = [tsample.mean];
            % error for multivariate case
            TV = [tsample.variance];
            this.t1moments = TM;
            this.t2moments = TV + TM.^2;
            
        end
        
        function mfz = apply_ep(this, mxf,   mzf)
            % Input: mzf (cavity) is DistNormal
            %   mxf:  GKConvolvable
            % Output: mfz is a DistNormal
            %
            X = this.f1sample;
            Z = this.f2sample;
            % ZT = Z target = z'
%             ZT = this.tsample;
            Mux = kerGGaussian( X, mxf, this.embed_width1, this.gw1);
            Muz = kerGGaussian( Z, mzf, this.embed_width2, this.gw2);
            
            % Operator application
            Alpha = this.operator*( Mux.*Muz );
        
            % projection
            mean = this.t1moments*Alpha;
            cov = this.t2moments*Alpha - mean*mean';
            
            Dis = DistNormal(mean, cov);
            % divide after projection
            mfz = Dis/mzf;
%                         mfz = Dis;
        end
        
        
    end %end methods
    
    methods (Static=true)
        
        function [Op] = learn_operator(X, Y, Z, o)
            % learn mean embedding operator taking 2 incoming messages
            % kerGGauss kernels. Do cross validation.
            % The operator can be thought of a mapping of messages x->f,
            % y->f into f->z where f is the factor
            %
            if nargin < 4
                o = [];
            end
            
            C = cond_ggauss_cv2(X, Y, Z, o);
            
%             CondOpGGauss2(f1sample, f2sample, tsample, ...
%                 operator, embed_width1, gw1, embed_width2, gw2)
            
            % medx = pairwise median distance of X
            xembedw = C.bxembed_width;
            yembedw = C.byembed_width;
            Op = CondOpGGauss2(X, Y, Z, C.operator, xembedw, C.skx, ...
                yembedw, C.sky);
        end
        
    end
    
end

