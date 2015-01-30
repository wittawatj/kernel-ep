classdef DistNormal < handle & GKConvolvable & Sampler ...
        & Density & Distribution & HasHellingerDistance & HasKLDivergence ...
        & PrimitiveSerializable
    %DIST_NORMAL Gaussian distribution object for kernel EP framework.
    
    properties (SetAccess=protected)
        mean
        % precision matrix
        precision
        d %dimension
        variance=[];
        
        % from Distribution
        parameters;
    end
    
    properties (SetAccess=protected, GetAccess=protected)
        Z % normalization constant
    end
    
    
    methods
        %constructor
        function this = DistNormal(m, variance)
            assert(~isempty(m));
            assert(~isempty(variance));
            if size(m, 1)==1 && size(variance, 1)==1 && size(m, 2) > 1
                % object array of many 1d Gaussians
                assert(size(m, 2)==size(variance, 2), '1d Gaussians: #means and #variances do not match');
                n = size(m, 2);
                this = DistNormal.empty();
                for i=1:n
                    this(i) = DistNormal(m(i), variance(i));
                end
            elseif size(m, 1)>1 && size(m, 2)>1
                % object array of many multivariate Gaussians
                assert(ndims(variance)==3, ['multidimensional means must be '...
                    'accompanied with 3-d variance variable.']);
                assert(size(variance,3)==size(m, 2), '#means and #variances do not match');
                n = size(m, 2);
                this = DistNormal.empty();
                for i=1:n
                    this(i) = DistNormal(m(:, i), variance(:, :, i));
                end
            else
                % one object, any dimension
                this.mean = m(:);
                this.d = length(this.mean);
                assert(all(size(variance)==size(variance'))) %square
                this.variance = variance;
                this.parameters = {this.mean, this.variance};
            end
            
        end
        
        function prec = get.precision(this)
            if isempty(this.precision)
                % expensive. Try to find a way for lazy evaluation later.
%                 reg = (abs(this.variance) < 1e-5)*1e-5;
                if isscalar(this.variance)
                    this.precision = 1/this.variance;
                else
                    this.precision = inv(this.variance );
                end
                
            end
            prec = this.precision;
        end
        
         
        function X = draw(this, N)
            % return dxN sample from the distribution
            X = mvnrnd(this.mean', this.variance, N)';
            
        end
        
        function Mux = conv_gaussian(this, X, gw)
            % X (dxn)
            % gw= a scalar for Gaussian kernel parameter
            % convolve this distribution (i.e., a message) with a Gaussian
            % kernel on sample in X. This is equivalent to an expectation of
            % the Gaussian kernel with respect to this distribution
            % (message m): E_{m(Y)}[k(x_i, Y] where x_i is in X
            [d,n] = size(X);
            assert(d==length(this.mean));
            
            %             we can do better. sqrt(det(2*pi*Sigma)) will cancel anyway.
            Sigma = gw*eye(d);
            Mux = sqrt(det(2*pi*Sigma))*mvnpdf(X', this.mean(:)', this.variance+ Sigma)';  
            
        end
        
        function D=density(this, X)
            
            % Variance can be negative in EP. mvnpdf does not accept it.
            %             D = mvnpdf(X', this.mean(:)', this.variance + 1e-6*eye(d) )';
            
            % Naive implementation. Can do better with det(.) ?
            P = this.precision;
            PX = P*X;
            mu = this.mean;
            I = 0.5*( sum(X.*PX, 1) + mu'*P*mu - 2*mu'*PX );
            D = this.Z*exp(-I);
        end
        
        function f=func(this)
            % return a function handle for density. Useful for plotting
            f = @(x)mvnpdf(x, this.mean, this.variance);
        end
        
        function z = get.Z(this)
            % Z is used with multiplication not division.
            if isempty(this.Z)
                d = length(this.mean);
                this.Z = ((2*pi)^(-d/2))*(det(this.variance)^(-1/2));
            end
            z = this.Z;
        end
        
        function p=isProper(this)
            % return true if this is a proper distribution e.g., not have
            % negative variance.

            mm = norm(this.mean);
            vv = norm(this.variance, 'fro');
            if this.d == 1
                p = isfinite(vv) && isfinite(mm) && this.variance >0;
            else 
                % multivariate Gaussian
                p = isfinite(vv) && isfinite(mm) && all(eig(this.variance)>0);

            end
        end

        function t=getDistType(this)
            t = mfilename;
        end

        function names=getParamNames(this)
            names={'mean', 'variance'};
        end
        
        function dist=distHellinger(this, d2)
            % Compute Hellinger distance from this DistNormal to d2,
            % another DistNormal. Hellinger distance is bounded between 0
            % and 1
            % Refer: https://en.wikipedia.org/wiki/Hellinger_distance
            
            assert(isa(d2, 'DistNormal'));
            assert(this.d==1, 'Hellinger distance is for d=1 presently.');
            m1 = this.mean;
            v1 = this.variance;
            m2 = d2.mean;
            v2 = d2.variance;
            c = sqrt(2*sqrt(v1)*sqrt(v2)/(v1+v2 ));
            e = exp(-(1/4)*(m1-m2)^2/(v1+v2) );
            dist = sqrt(1- c*e );
            
        end
        
        function div=klDivergence(this, d2)

            assert(this.d==d2.d, 'The two dists must have the same dimension to compute KL');
            assert(this.isProper(), 'This DistNormal is not proper.')
            assert(d2.isProper(), 'd2 is not a proper DistNormal');

            if this.d==1
                v1 = this.variance;
                v2 = d2.variance;
                m1 = this.mean;
                m2 = d2.mean;

                div = ( v1/v2 + ((m1-m2)^2)/v2 -1 -log(v1/v2) )/2.0;
            else
                % multivariate case 
                m2_m1 = d2.mean - this.mean;
                V1 = this.variance;
                P2 = d2.precision;
                div = 0.5*(P2(:)'*V1(:)+ m2_m1'*P2*m2_m1 - this.d - log(det(V1)*det(P2)));
            end

        end

        function X = sampling0(this, N)
            X = this.draw( N);
        end

        function D = mtimes(this, distNorm)
            if ~isa(distNorm, 'DistNormal')
                error('mtimes only works with DistNormal obj.');
            end
            m1 = this.mean;
            p1 = this.precision;
            m2 = distNorm.mean;
            p2 = distNorm.precision;

            prec = p1+p2;
            nmean = prec \ (p1*m1 + p2*m2);
            %  bad idea to invert ?
            var = inv(prec);
            D = DistNormal(nmean, var);
        end

        function D = mrdivide(this, distNorm)
            if ~isa(distNorm, 'DistNormal')
                error('mrdivide only works with DistNormal obj.');
            end
            m1 = this.mean;
            p1 = this.precision;
            m2 = distNorm.mean;
            p2 = distNorm.precision;

            % create a problem if p1=p2 ***
            prec = p1-p2;
            nmean = prec \ (p1*m1 - p2*m2);
            %  bad idea to invert ?
            var = inv(prec);
            D = DistNormal(nmean, var);
        end

        %%%%%%%%%%%%%%%%%%%%5
        function s=saveobj(this)
            s.mean=this.mean;
            s.variance=this.variance;
        end

        % from PrimitiveSerializable interface
        function s=toStruct(this)
            %mean
            %% precision matrix
            %precision
            %d %dimension
            %variance=[];
            %% from Distribution
            %parameters;
            s = struct();
            s.className=class(this);
            % mean can be a vector
            s.mean = this.mean;
            %s.d = this.d;
            s.variance = this.variance; 
            %s.precision = this.precision;
        end
    end %end methods


    methods (Static)
        function obj=loadobj(s)
            obj=DistNormal(s.mean, s.variance);
        end

        function S=suffStat(X)
            ssb = DistNormalBuilder();
            S = ssb.suffStat(X);
        end

        function D=fromSuffStat(S)
            ssb = DistNormalBuilder();
            D = ssb.fromSuffStat(S);
        end

        function ssb=getDistBuilder()
            ssb = DistNormalBuilder();
        end

    end %end static methods
end
