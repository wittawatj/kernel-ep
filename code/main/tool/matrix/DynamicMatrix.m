classdef DynamicMatrix < handle
    %DYNAMICMATRIX A matrix which does not store all the entries but dynamically
    %generate entries when need. 
    %   Entries are generated from the specified generator function. 
    %   The point of this matrix is to save memory.
    %
    %   For documentation purpose, assume this object = M of size dxn.
    %   
    
    properties(Abstract, SetAccess=protected)
        % Return the generator function associated with this DynamicMatrix.
        % A generator is a function handle f: (I, J) -> M where I,J are index list
        % and M is a submatrix specified by I, J.
        generator;

    end
    
    methods (Abstract)
        
        % override to allow indexing with M(:, 2:5) for example.
        % assert(isnumeric(B));
        % This is not a good idea since it will also override . operator for 
        % method calls.
        % B = subsref(this, S);

        i = end(this, dim, totalDim);

        % just like size(M, dim)
        s = size(this, dim);
        
        % S = M*M'.  S is square and symmetric.
        S = mmt(this);

        % assume this=M is (dxN). 
        % Compute the diagonal entries of M'(MM'+ lambda*I)^-1 * M .
        % This method is a bit specific to LOOCV for ridge regression solutions.
        % d = diag, m=M, t=transpose, i=inverse
        % Input: mmt = M*M'. If not specified, it should be computed by 
        % this.mmt()
        % return a row vector of length this.cols.
        v = dmtim(this, lambda, mmt);

        % convert back to numerical matrix 
        M = toNumeric(this);

        % compute M*R. Ideally this should work with DynamicMatrix R.
        B = rmult(this, R);

        % compute L*M
        B = lmult(this, L);

        % access this(I, J) as if (this) is a numerical matrix.
        S = index(this, I, J);

        % access row i. R is a row vector.
        R = row(this, i);

        % access column j. C is a column vector.
        C = col(this, j);


    end
    

end

