classdef MatUtils < handle
    %MATUTILS Class containing convenient methods for manipulating matrices
    
    properties
    end
    
    methods(Static)
        function Z=colKronecker(X, Y)
            % Kronecker product for columns in X, Y
            % See also section 10.1.6 of Matlab array manipulation tips and 
            % tricks by Peter J. Acklam
            %
            assert(isnumeric(X));
            assert(isnumeric(Y));
            assert(size(X, 2)==size(Y, 2), 'X and Y must have the same #columns');
            rx=size(X, 1);
            ry=size(Y, 1);

            %I=repmat(1:ry, rx, 1);
            %Z=repmat(X, ry, 1).*Y(I, :);
            
            % Behave like a Kronecker product for column vectors.
            I=repmat(1:rx, ry, 1);
            Z=X(I, :).*repmat(Y, rx, 1);
        end

        function O=colOutputProduct(X, Y)

            % Compute the output product of each column in X and Y
            % If X is rx x c and Y is ry x c, Z is rx x ry x c.
            assert(isnumeric(X));
            assert(isnumeric(Y));
            assert(size(X, 2)==size(Y, 2), 'X and Y must have the same #columns');
            c = size(X, 2);
            O = MatUtils.colKronecker(Y, X);
            O = reshape(O, [size(X,1), size(Y, 1), c]);
        end

        function v = quadraticFormAlong3Dim(V, Q)
            % Input:
            %  - V: d x m
            %  - Q: d x d x n 
            % Output:
            %  - v: v(i, j) = V(:, i)'*Q(:, :, j)*V(:, i), an m x n matrix
            %
            [d, m] = size(V);
            n = size(Q, 3);
            assert(size(Q, 1) == size(Q, 2), 'first two dimensions of Q must be the same.');
            assert(size(V, 1) == size(Q, 1), 'dim 1 of V must be the same as dim 1,2 of Q');
            left = V'*Q(:, :); % m x d*n
            left3d = reshape(left, [m, d, n]);
            v = zeros(m, n);
            for i=1:n
                v(:, i) = sum(left3d(:, :, i).*V', 2);
            end

        end

        function v = flattenCell(C)
            % Input: 
            %   - C: a cell array where C{i} is a vector (row or column)
            % Output: 
            %   - a row vector obtained by concatenating all vectors in C
            % 
            C2 = cellfun(@(v)v(:)', C, 'UniformOutput', false);
            v = [C2{:}];
        end

        function C = partitionArray(v, partition)
            % Input: 
            %   - v: an array 
            %   - partition: an array of positive integers that sums to length(v). 
            %   This specifies the size of each partition.
            % Output: 
            %   - a cell array of length = length(partition) and C{i} = partition 
            %   i of v in a row vector.
            %
            assert(all(partition) > 0, 'partition size must be > 0');
            assert(isnumeric(v));
            assert(sum(partition) == length(v), 'sum of partition must be length(v)');
            C = cell(1, length(partition));
            from = 1;
            for i=1:length(C)
                par_size_i = partition(i);
                C{i} = v(from:(from+par_size_i-1));
                from = from + par_size_i;
            end
        end
    end

end

