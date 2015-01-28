classdef IncompChol < handle
    %INCOMPCHOL A class allowing an easy and extensible computation of
    %incomplete Cholesky.
    %
    
    properties (SetAccess=protected)
        % reduced data. Immutable. incomplete Cholesky does not need full
        % data to compute new column in R for a new point. 
        % RX is a reduced dataset (data inside Instances).
        RX;
        % handle to a kernel function
        kfunc;
        % tolerance level in incomplete Cholesky
        tol;
        
        % Cholesky matrix such that full kernel matrix K (nxn) is approximately
        % R'*R where R is rank(R)xn and rank(R) << n.
        R;
        
        % vector 1xrank(R) needed for computing a Cholesky feature vector
        % for a new point.
        nu;
        
        % Column indices of X to keep.
        keepI;
    end
    
    
    methods
        
        function this=IncompChol(X, kfunc, tol, maxrank)
            % - X is an Instances.
            % - kfunc is a Kernel taking two instances in X and
            % compute their inner product. 
            
            assert(isa(X, 'Instances'));
            assert(isa(kfunc, 'Kernel'));
            this.kfunc = kfunc;
            this.tol = tol;
            if nargin < 4 
                maxrank = X.count();
            elseif maxrank <=0
                maxrank = X.count();
            end
            % Do Cholesky now. Data is immutable.
            [R, I, nu] = IncompChol.incomp_chol(X, kfunc, tol, maxrank);
            
            this.R = R;
            % Reduce X.
            this.RX = X.get(I);
            this.nu = nu;
            this.keepI = I;
        end
        
        function Rt=chol_project(this, Xtest)
            % Compute a Cholesky feature vectors for new points in Xtest.
            % Assume Xtest has M instances. Then, Rt is rank(R)xM.
            assert(isa(Xtest, 'Instances'));
            R = this.R;
            nu = this.nu;
            
            M = Xtest.count();
            ra = size(R, 1); %rank
            Rt = zeros(ra, M);
            Kt = this.kfunc.eval(this.RX, Xtest.getAll());
            assert(all(size(Kt)==[ra, M]));
            % reduced R. Should I cache RR ?
            RR = this.R(:, this.keepI);
            % parfor is possible for m
            for m=1:M
                r = zeros(ra, 1);
                % Does not need all entries in k
                k = Kt(:, m);
                for j=1:ra
                    r(j) = ( k(j) -r'*RR(:,j) )/nu(j);
                end
                Rt(:, m) = r;
            end
            
        end
        
        
    end %end methods
    
    methods (Static)
        
        function [R, I, nu] = incomp_chol(X, kfunc, eta, maxrank)
            % Perform incomplete Cholesky factorization on the full kernel
            % matrix K using the tolerance level (tol) eta. The factorization
            % is such that K is approximately R'*R.
            % - eta gives threshold residual cutoff
            % Return R = new features stored in matrix R of size T x ell
            %
            assert(isa(X, 'Instances'));
            assert(isa(kfunc, 'Kernel'));
            assert(eta>=0, 'Threshold residual cutoff must be non-negative');
            ell = X.count();
            if nargin < 4 
                maxrank = ell;
            end
            maxrank = max(0, min(maxrank, ell) );
            j = 0;
            R = zeros(maxrank, ell);
%             R = gpuArray(R);
            Xdat = X.getAll();
            % compute diagonal entries of K
            d = kfunc.pairEval(Xdat, Xdat);
            d = d(:)'; % make it a row vector
%             d = diag(K);
            [a, I(j+1)] = max(d);
            if a <= eta
                error('eta too high. Should be: eta < max(diag(K)).');
            end
            
            % Ind = [];
%             allI = 1:ell;
%             nu = zeros(1, ell);
            while a > eta
                j = j+1;
%                 beforeIj = allI<=I(j);
                beforeIj = 1:I(j);
%                 afterIj = ~beforeIj;
                afterIj = (I(j)+1):ell;
                nu(j) = sqrt(a);
                % row I(j) of K
                k_Ij = kfunc.eval( X.get(I(j)),  X.getAll() );
                assert( length(k_Ij)==X.count());
                R(j, beforeIj) = ( k_Ij(beforeIj) - R(:, I(j))'*R(:, beforeIj) )/nu(j);
                R(j, afterIj) = ( k_Ij(afterIj) - R(:, I(j))'*R(:, afterIj) )/nu(j);
                
                d = d - R(j, :).^2;
                [a, I(j+1)] = max(d);
                if j >= min(maxrank, ell)
                    % maxrank reached. stop
                    break;
                end
            end
            T = j;
            R = R(1:T, :);
            I = I(1:T);
%             nu = nu(1:T);
            %display(sprintf('IncompChol returns with rank %d', T));
        end
        
    end
end

