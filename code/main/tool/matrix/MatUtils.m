classdef MatUtils < handle
    %MATUTILS Class containing convenient methods for manipulating matrices
    
    properties
    end
    
    methods(Static)
        function Z=colOutputProduct(X, Y)
            % Compute the output product of each column in X and Y
            % If X is rx x c and Y is ry x c, Z is rx*ry x c.
            %
            % See also section 10.1.6 of Matlab array manipulation tips and 
            % tricks by Peter J. Acklam
            %
            assert(isnumeric(X));
            assert(isnumeric(Y));
            assert(size(X, 2)==size(Y, 2), 'X and Y must have the same #columns');
            rx=size(X, 1);
            ry=size(Y, 1);

            I=repmat(1:ry, rx, 1);
            Z=repmat(X, ry, 1).*Y(I, :);
        end
    end

end

