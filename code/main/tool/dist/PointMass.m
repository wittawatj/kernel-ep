classdef PointMass < handle & GKConvolvable & Density
    %POINTMASS Point mass distribution. Useful for representing evidence.
    %Evidence is equivalent to having a point mass distribution over the
    %observed values. By treating evidence as a type of distribution, we do
    %not have to treat an evidence in a special way.
    
    properties (SetAccess=private)
        point;
    end
    
    methods
        function this=PointMass(point)
            this.point = point;
        end
        
        function Mux = conv_gaussian(this, X, gw)
            % X (dxn)
            % Sigma (dxd) or a scalar (assume Sigma*I in that case) =
            % kernel parameter
            % convolve this distribution (i.e., a message) with a Gaussian
            % kernel on sample in X. This is equivalent to an expectation of
            % the Gaussian kernel with respect to this distribution
            % (message m): E_{m(Y)}[k(x_i, Y] where x_i is in X
            
            Mux = kerGaussian(X, this.point, gw)';
        end
        
        function D=density(this, X)
            [d,n]=size(X);
            D = all(repmat(this.point, 1, n) == X, 1);
        end
    end %end methods
    
end

