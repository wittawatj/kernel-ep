classdef CondOp1 < handle
    %CONDOP1 Conditional mean embedding operator taking one input. Assume
    %the output suff. stat. mapping is finite-dimensional phi(z)=[z, z^2]
    %for Gaussian distribution. C_{z|x)
    
    properties (SetAccess=private)
        fsample %X
        tsample %Z
        gw % Gaussian kernel width
        lambda % regularization parameter
        ker_mat %kernel matrix
        operator % operator
    end
    
    methods
        % f = from
        % t = to
        % gw = Gaussian width of kernel associated with this operator
        % regularization parameter
        function this = CondOp1(fsample, tsample, gw, lamb)
            this.fsample = fsample;
            this.tsample = tsample;
            this.gw = gw;
            this.lambda = lamb;
        end
        
        function op = get.operator(this)
            if isempty(this.operator)
                [d,n]=size(this.fsample);
                this.operator = inv(this.ker_mat + this.lambda*eye(n));
            end
            op = this.operator;
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
%             Beta = real( mzf.density(Z)' );
%             C = Alpha.*Beta;
            C = Alpha;
            S = DistNormal.normalSuffStat(Z);
            % projection
            suffStat = S*C;
            [dz,nz] = size(Z);
            mean = suffStat(1:d);
            cov = reshape(suffStat( (d+1):end), [dz,dz]) - mean*mean'; 
            
            Dis = DistNormal(mean, cov);
            % divide after projection
%             mfz = Dis/mzf;
            mfz = Dis;
            
        end
        
        function K = get.ker_mat(this)
            if isempty(this.ker_mat)
                % Kernel matrix
                X = this.fsample;
                Kx = kerGaussian(X, X, this.gw);
                this.ker_mat = Kx;
                
            end
            K = this.ker_mat;
        end
    end %end methods
    
    methods (Static=true)
        
        function [Op] = learn_operator(X, Z_X, o)
            % learn mean embedding operator taking 1 input
            % Gaussian kernels. Do cross validation.
            % The operator can be thought of a mapping of messages x->f
            % into f->z where the factor f is represented by the conditional sample of
            % Z_X on X
            %
            if nargin <3
                o = [];
            end
 
            C = cond_embed_cv1(X, Z_X, o);
            
            % medx = pairwise median distance of X
            skx = C.bxw * C.medx; %bxw = best Gaussian width for x
            lamb = C.blambda;
            
            Op = CondOp1(X, Z_X, skx, lamb);
        end
        
    end
    
end

