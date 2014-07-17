classdef DefaultDynamicMatrix < DynamicMatrix
    %DEFAULTDYNAMICMATRIX Default implementation of DynamicMatrix.
    %   Detailed explanation goes here
    
    properties (SetAccess=protected)
        % A generator is a function handle f: (I, J) -> M where I,J are index list
        % and M is a submatrix specified by I, J.
        generator;
        % Fixed number of rows
        rows;
        % Fixed number of columns
        cols;

    end

    properties 
        % roughly handle (chunkSize) elements at a time
        chunkSize;
    end

    
    methods

        function this=DefaultDynamicMatrix(generator, rows, cols )
            % chunkSize is optional.
            assert(isa(generator, 'function_handle'));
            assert(rows > 0);
            assert(cols > 0);

            this.generator = generator;
            this.rows = rows;
            this.cols = cols;
                % number of elements to process at a time
                % 1e6 -> 8MB
            this.chunkSize = 2e6;
        end


        % override to allow indexing with M(:, 2:5) for example.
        % This interferes with the usual method calls.
        % this.method() will also trigger subsref(). 
        %
        % assert(isnumeric(B));
        %function B = subsref(this, S)
        %    if ~strcmp(S.type, '()')
        %        error('DefaultDynamicMatrix only supports indexing with ().');
        %    end
        %    subs = S.subs;
        %    assert(iscell(subs));
        %    if length(subs)==1
        %        error('Only 2d indexing is allowed.');
        %    end
        %    I = subs{1};
        %    if ischar(I) 
        %        assert(strcmp(I, ':'), 'I is %s which is not :', I);
        %        I = 1:this.rows;
        %    end
        %    J = subs{2};
        %    if ischar(J) 
        %        assert(strcmp(J, ':'), 'J is %s which is not :', J);
        %        J = 1:this.cols;
        %    end
        %    g = this.generator;
        %    B = g(I, J);
        %    assert(size(B,1)==length(I), 'unexpected #rows returned by generator');
        %    assert(size(B,2)==length(J), 'unexpected #cols returned by generator'); 
        %end
            
        function set.chunkSize(this, chunkSize)

            assert(chunkSize>0);
            this.chunkSize = chunkSize;
        end

        function i = end(this, dim, totalDim)
            if totalDim ~= 2
                error('end can only appear on 2d indexing');
            end
            if dim==1
                i = this.rows;
            else
                i = this.cols;
            end
                
        end
        
        % just like size(M, dim)
        function varargout = size(this, dim)
            l = [this.rows, this.cols];
            if nargin < 2
                if nargout < 2 
                    varargout = {l};
                elseif nargout==2
                    varargout{1} = l(1);
                    varargout{2} = l(2);
                else
                    error('There are at most 2 outputs from size().')
                end
                return;
            end

            if ~(dim==1 || dim==2) 
                error('invalid dimension');
            end
            varargout = {l(dim)};
        end

        % B = M*M'. B is square and symmetric.
        function B = mmt(this) 
            r = this.rows;
            c = this.cols;
            chunkCols = floor(this.chunkSize/r);
            chunkCols = max(1, chunkCols);
            B = zeros(r, r);
            colStart = 1;
            g = this.generator;
            %display(sprintf('chunkCols: %d', chunkCols));
            while colStart <= c
                colEnd = min(colStart+chunkCols-1, c);
                colRange = colStart:colEnd;
                subM = g(1:r, colRange);
                B = B + subM*subM';

                colStart = colEnd + 1;
            end
        end

        function v = dmtim(this, lambda, mmt)
            if nargin < 3
                mmt = this.mmt();
            end
            D = size(mmt, 1);
            assert(D==this.rows);
            % such that R'*R = A
            A = mmt + lambda*eye(D);
            R = chol(A);
            clear A;

            chunkCols = floor(this.chunkSize/this.rows);
            chunkCols = max(1, chunkCols);
            %display(sprintf('chunkCols: %d', chunkCols));
            v = zeros(1, this.cols);
            colStart = 1;
            g = this.generator;
            % linsolve options 
            ltopts = struct('LT', true);
            utopts = struct('UT', true);
            while colStart <= this.cols 
                colEnd = min(colStart+chunkCols-1, this.cols);
                colRange = colStart:colEnd;
                subM = g(1:this.rows, colRange);
                %y = R'\subM;
                y = linsolve(R', subM, ltopts);

                %X = R\y; % D x chunkCols
                X = linsolve(R, y, utopts);
                v(colRange) = sum(subM.*X, 1);
                
                colStart = colEnd + 1;
            end
            

        end

        % convert back to numerical matrix 
        function M = toNumeric(this)
            g = this.generator;
            M = g(1:this.rows, 1:this.cols);
        end

        % compute M*R. Ideally this should work with DynamicMatrix R.
        function B = rmult(this, R)
            [r2, c2] = size(R);
            if this.cols ~= r2
                error('#rows of R does not match #cols of this');
            end
            % work on (chunkCols) columns at a time
            chunkCols = floor(this.chunkSize/this.rows);
            chunkCols = max(1, chunkCols);
            B = zeros(this.rows, c2);
            colStart = 1;
            g = this.generator;
            while colStart <= this.cols
                colEnd = min(colStart+chunkCols-1, this.cols);
                colRange = colStart:colEnd;
                subM = g(1:this.rows, colRange);
                %subR = R(colRange, 1:c2);
                subR = DefaultDynamicMatrix.subIndex(R, colRange, 1:c2);
                B = B + subM*subR;

                colStart = colEnd + 1;
            end
        end

        % compute L*M
        function B = lmult(this, L)
            [r2, c2] = size(L);
            if c2 ~= this.rows
                error('#cols of L does not match #rows of this');
            end
            % work on (chunkRows) rows of this at a time
            chunkRows = floor(this.chunkSize/this.cols);
            chunkRows = max(1, chunkRows);
            B = zeros(r2, this.cols);
            rowStart = 1;
            g = this.generator;
            while rowStart <= this.rows
                rowEnd = min(rowStart+chunkRows-1, this.rows);
                rowRange = rowStart:rowEnd;
                subM = g(rowRange, 1:this.cols);
                %subL = L(1:r2, rowRange);
                subL = DefaultDynamicMatrix.subIndex(L, 1:r2, rowRange);
                B = B+subL*subM;

                rowStart = rowEnd + 1;

            end
        end

        function S = index(this, I, J)
            g = this.generator;
            S = g(I, J);
        end
        
        % access row i. R is a row vector.
        function R = row(this, i)
            g = this.generator;
            J = 1:this.cols;
            R = g(i, J);
        end


        % access column j. C is a column vector.
        function C = col(this, j)
            g = this.generator;
            I = 1:this.rows;
            C = g(I, j);
        end

    end %end methods

    methods (Static)
        function dm=fromMatrix(mat)
            % Return a DefaultDynamicMatrix for the specified matrix mat.
            % dm = dynamic matrix
            assert(isnumeric(mat));
            [r, c] = size(mat);
            gen = @(I,J)mat(I, J);
            dm = DefaultDynamicMatrix(gen, r, c);

        end

        function S=subIndex(M, I, J)
            %  Return M(I, J). Also work if M is a DynamicMatrix
            assert(isa(M, 'DynamicMatrix') || isnumeric(M));
            if isnumeric(M)
                S = M(I, J);
            elseif isa(M, 'DynamicMatrix')
                S = M.index(I, J);
            else
                error('M has an unsupported type');
            end

        end

    end

    
end

