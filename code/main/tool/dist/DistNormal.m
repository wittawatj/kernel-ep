classdef DistNormal < handle & GKConvolvable & Sampler & Density & Distribution
    %DIST_NORMAL Gaussian distribution object for kernel EP framework.
    
    properties (SetAccess=private)
        mean
        % precision matrix
        precision
        d %dimension
        variance=[];
    end
    
    properties (SetAccess=protected, GetAccess=protected)
        Z % normalization constant
    end
    
    
    methods
        %constructor
        function this = DistNormal(m, var)
            assert(~isempty(m));
            assert(~isempty(var));
            if size(m, 1)==1 && size(var, 1)==1 && size(m, 2) > 1
                % object array of many 1d Gaussians
                assert(size(m, 2)==size(var, 2));
                n = size(m, 2);
                this = DistNormal.empty();
                for i=1:n
                    this(i) = DistNormal(m(i), var(i));
                end
            else
                % one object
                this.mean = m(:);
                this.d = length(this.mean);
                assert(all(size(var)==size(var'))) %square
                this.variance = var;
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
        
        function p=isproper(this)
            % return true if this is a proper distribution e.g., not have
            % negative variance.
            if length(this.mean) > 1
                error('Not yet supported for dim>1');
            end
            vv = norm(this.variance, 'fro');
            mm = norm(this.mean);
            p = isfinite(vv) && isfinite(mm) && this.variance >0;
        end
        
        function dist=distHellinger(this, d2)
            % Compute Hellinger distance from this DistNormal to d2,
            % another DistNormal. Hellinger distance is bounded between 0
            % and 1
            % Refer: https://en.wikipedia.org/wiki/Hellinger_distance
            assert(this.d==1, 'Hellinger distance is for d=1 presently.');
            assert(isa(d2, 'DistNormal'));
            m1 = this.mean;
            v1 = this.variance;
            m2 = d2.mean;
            v2 = d2.variance;
            c = sqrt(2*sqrt(v1)*sqrt(v2)/(v1+v2 ));
            e = exp(-(1/4)*(m1-m2)^2/(v1+v2) );
            dist = sqrt(1- c*e );
            
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
        
    end %end methods
    
    
    methods (Static)
        
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
