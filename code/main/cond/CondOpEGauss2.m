classdef CondOpEGauss2 < handle
    %CONDOPEGAUSS2 Conditional mean embedding operator taking 2 messages
    %(distributions) and producing one outgoing message.
    %   The outgoing message is a Gaussian computed by a finite-dimensional
    %   feature map of the form (z, z^2).
    % Conditional mean embedding operator: C_{z'|xz}
    
    properties (SetAccess=private)
        f1sample %X. f1 = from the first
        f2sample %Y
        tsample %Z
        gw1 % Gaussian kernel width for f1sample
        gw2 % Gaussian kernel width for f2sample
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
        function this = CondOpEGauss2(f1sample, f2sample, tsample, ...
                operator, gw1, gw2)
            assert(size(f1sample, 2) == size(f2sample, 2));
            assert(size(f2sample, 2) == size(tsample, 2));
            assert(isa(f1sample, 'DistNormal'));
            assert(isa(f2sample, 'DistNormal'));
            assert(isa(tsample, 'DistNormal'));
            assert(gw1 > 0);
            assert(gw2 > 0);
            
            this.f1sample = f1sample;
            this.f2sample = f2sample;
            this.tsample = tsample;
            this.gw1 = gw1;
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
            %
 
            X = this.f1sample;
            Z = this.f2sample;
            % ZT = Z target = z'
%             ZT = this.tsample;
            Mux = kerEGaussian( X, mxf, this.gw1);
            Muz = kerEGaussian( Z, mzf, this.gw2);
            
            % Operator application
            Alpha = this.operator*( Mux.*Muz );
        
            % projection
            mean = this.t1moments*Alpha;
            cov = this.t2moments*Alpha - mean*mean';
            
            Dis = DistNormal(mean, cov);
            % divide after projection
            mfz = Dis/mzf;
            %             mfz = Dis;
        end
        
        
    end %end methods
    
    methods (Static=true)
        
        function [Op] = learn_operator(X, Y, Z, o)
            % learn mean embedding operator taking 2 incoming messagesl
            % EGauss kernels. Do cross validation.
            % The operator can be thought of a mapping of messages x->f,
            % y->f into f->z where f is the factor
            %
            if nargin < 4
                o = [];
            end
            
            C = cond_egauss_cv2(X, Y, Z, o);
            
            % medx = pairwise median distance of X
            skx = C.bxw * C.medx; %bxw = best Gaussian width for x
            sky = C.byw * C.medy;
      
            Op = CondOpEGauss2(X, Y, Z, C.operator, skx, sky);
        end
        
    end
    
end

