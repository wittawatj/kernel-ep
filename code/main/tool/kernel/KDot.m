classdef KDot < Kernel
    %KDOT A kernel for standard Euclidean dot product
    
    properties
    end
    
    methods
        function this=KDot()
            
        end
        
        function Kmat = eval(this, X, Y)
            % X, Y are data matrices where each column is one instance
            assert(isnumeric(X));
            assert(isnumeric(Y));
            Kmat = X'*Y;
            
        end
        
        function Kvec = pairEval(this, X, Y)
            
            assert(isnumeric(X));
            assert(isnumeric(Y));
            n1=size(X, 2);
            n2=size(Y, 2);
            assert(n1==n2);
            Kvec = sum(X.*Y, 1);
            
        end
        
        function Param = getParam(this)
            Param = {};
        end
        
        function s=shortSummary(this)
            s = 'KDot';
        end
    end
    
end

