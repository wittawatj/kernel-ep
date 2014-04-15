classdef CondOp2 < handle
    %CONDOP2 Conditional mean embedding operator taking two inputs. Assume
    %the output suff. stat. mapping is finite-dimensional phi(z)=[z, z^2]
    %for Gaussian distribution. C_{z|x,y}
    
    properties (SetAccess=private)
        f1sample %X. f1 = from the first
        f2sample %Y
        tsample %Z
        gw1 % Gaussian kernel width for f1sample
        gw2 % Gaussian kernel width for f2sample
        
        lambda % regularization parameter for operat
        
        ker_mat1 %kernel matrix on f1sample
        ker_mat2
        operator % operator = (K1.^K2 + lambda*I)^-1
    end
    
    methods
        % f = from
        % t = to
        % gw1, gw2 = Gaussian width of kernel associated with this operator
        % lamb = regularization parameter
        function this = CondOp2(f1sample, f2sample, tsample, gw1, gw2, lamb)
            assert(size(f1sample, 2) == size(f2sample, 2));
            assert(size(f2sample, 2) == size(tsample, 2));
            assert(gw1 > 0);
            assert(gw2 > 0);
            assert(lamb >= 0);
            
            this.f1sample = f1sample;
            this.f2sample = f2sample;
            this.tsample = tsample;
            this.gw1 = gw1;
            this.gw2 = gw2;
            this.lambda = lamb;
        end
        
        function op = get.operator(this)
            if isempty(this.operator)
                [~,n] = size(this.f1sample);
                K1 = this.ker_mat1;
                K2 = this.ker_mat2;
                % good idea to find the inverse ?? Maybe better with
                % Cholesky ?
                this.operator = inv(K1.*K2 + this.lambda*eye(n));
            end
            op = this.operator;
        end
        
        function K = get.ker_mat1(this)
            if isempty(this.ker_mat1)
                % Kernel matrix
                X1 = this.f1sample;
                K1 = kerGaussian(X1, X1, this.gw1);
                this.ker_mat1 = K1;
                
            end
            K = this.ker_mat1;
        end
        
        function K = get.ker_mat2(this)
            if isempty(this.ker_mat2)
                % Kernel matrix
                X2 = this.f2sample;
                K2 = kerGaussian(X2, X2, this.gw2);
                this.ker_mat2 = K2;
                
            end
            K = this.ker_mat2;
        end
        
        function mfz = apply_pbp(this, mxf, myf)
            % Apply this operator to two incoming messages. Used for projected BP
            
            assert(isa(mxf, 'GKConvolvable'));
            assert(isa(myf, 'GKConvolvable'));
            
            X = this.f1sample;
            Y = this.f2sample;
            Z = this.tsample;
            
            Mux = mxf.conv_gaussian( X, this.gw1);
            Muy = myf.conv_gaussian( Y, this.gw2 );
            
            % Operator application
            Mup = Mux.*Muy;
            Alpha = this.operator*Mup(:);
            C = Alpha;
            S = DistNormal.normalSuffStat(Z);
            % projection
            suffStat = S*C;
            [dz,nz] = size(Z);
            mean = suffStat(1:dz);
            cov = reshape(suffStat( (dz+1):end), [dz,dz]) - mean*mean' ;
            
            mfz = DistNormal(mean, cov);
        end
        
    end %end methods
    
    methods (Static=true)
        
        function [Op] = learn_operator(X, Y, Z, o)
            % learn mean embedding operator taking 2 inputs
            % Gaussian kernels. Do cross validation.
            % The operator can be thought of a mapping of messages x->f,
            % y->f into f->z where f is the factor
            %
            if nargin < 4
                o = [];
            end
            
            C = cond_embed_cv2(X, Y, Z, o);
            
            % medx = pairwise median distance of X
            skx = C.bxw * C.medx; %bxw = best Gaussian width for x
            sky = C.byw * C.medy;
            lamb = C.blambda;
            
%             CondOp2(f1sample, f2sample, tsample, gw1, gw2, lamb)
            Op = CondOp2(X, Y, Z, skx, sky, lamb);
            
        end
        
    end
    
end

