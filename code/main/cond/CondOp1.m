classdef CondOp1 < handle
    %CONDOP1 Conditional mean embedding operator taking one input. Assume
    %the output suff. stat. mapping is finite-dimensional phi(z)=[z, z^2]
    %for Gaussian distribution. C_{z|x}
    
    properties (SetAccess=private)
        fsample %X
        tsample %Z
        
        operator % operator
        gw % Gaussian width
    end
    
    methods
        % f = from
        % t = to
        % gw = Gaussian width of kernel associated with this operator
        % regularization parameter
        function this = CondOp1(fsample, tsample, operator, gw)
            this.fsample = fsample;
            this.tsample = tsample;
            this.operator = operator;
            this.gw = gw;
        end
        
        
        function mfz = apply_bp(this, mxf)
            % Apply this operator to one incoming message.
            % Not for EP
            assert(isa(mxf, 'GKConvolvable'));
            X = this.fsample;
            Z = this.tsample;
            
            Mux = mxf.conv_gaussian( X, this.gw);
            [d,n] = size(X);
            % Operator application
            Alpha = this.operator*Mux(:);
            C = Alpha;
            S = DistNormal.suffStat(Z);
            % projection
            suffStat = S*C;
            [dz,nz] = size(Z);
            mean = suffStat(1:dz);
            cov = reshape(suffStat( (dz+1):end), [dz,dz]) - mean*mean' ;
            
            mfz = DistNormal(mean, cov);
        end
        
        function mfz = apply_ep_approx(this, mxf,   mzf)
            mfz = apply_ep(this, mxf,   mzf);
            
        end
        
        function mfz = apply_ep(this, mxf,   mzf)
            % Apply this operator to one incoming message for sending an EP message.
            % Let this operator be C_{z|x}
            % Compute m_{f->z} = proj[ f(x,z) m_{z->f} m_{x->f}]/m_{z->f}
            % where f is a factor.
            % We just need samples from f(x,z) (fsample, tsample)
            % Input: mzf (cavity) is DistNormal
            %   mxf:  GKConvolvable
            % Output: mfz is a DistNormal
            %
            %### EP projection should be done in a Factor. Fix later
            %
            assert(isa(mxf, 'GKConvolvable'));
            assert(isa(mzf, 'GKConvolvable'));
            X = this.fsample;
            Z = this.tsample;
            
            Mux = mxf.conv_gaussian( X, this.gw);
            [d,n] = size(X);
            % Operator application
            Alpha = this.operator*Mux(:);
            % Density can be imaginary because variance can be < 0
            % resulting in sqrt(det(variance)) being imaginary. Don't know
            % what to do here. Take real parts ??
            Beta = real( mzf.density(Z)' );
            
            C = Alpha.*Beta;
            %             C = Alpha;
            S = DistNormal.suffStat(Z);
            % projection
            suffStat = S*C/sum(C);
            [dz,nz] = size(Z);
            mean = suffStat(1:d);
            cov = reshape(suffStat( (d+1):end), [dz,dz]) - mean*mean' ;
            
            Dis = DistNormal(mean, cov);
            % divide after projection
            mfz = Dis/mzf;
            %             mfz = Dis;
        end
        
       
    end %end methods
    
    methods (Static=true)
        
        function [Op, CVLog] = learn_operator(X, Z_X, o)
            % learn C_{z|x}
            % Gaussian kernels. Do cross validation.
            % The operator can be thought of a mapping of messages x->f
            % into f->z where the factor f is represented by the conditional sample of
            % Z_X on X
            % It is assumed that (X, Z_X) ~ p(Z|X)s(X) where p(Z|X) is the
            % true distribution. s() can be arbitrary.
            %
            if nargin <3
                o = [];
            end
            
            CVLog = cond_embed_cv1(X, Z_X, o);
            skx = CVLog.bxw * CVLog.medx; %bxw = best Gaussian width for x
            Op = CondOp1(X, Z_X, CVLog.operator, skx);
        end
        
        function [Op, CVLog] = kbr_operator(X, Y, o)
            % Kernel Bayes rule operator. Learn C_{y|x}
            % learn mean embedding operator taking 1 input
            % Gaussian kernels. Do cross validation.
            % The operator C_{y|x} maps m_{x->f} to m_{f->y}. It is assumed
            % that (X,Y) ~ p(X|Y)s(Y) where p(X|Y) is the true
            % distribution. s() can be arbitrary.
            %
            if nargin <3
                o = [];
            end
            
            CVLog = kbr_cv1(X, Y,  o);
            skx = CVLog.bxw * CVLog.medx; %bxw = best Gaussian width for x
            Op = CondOp1(X, Y, CVLog.operator, skx);
        end
    end
    
end

