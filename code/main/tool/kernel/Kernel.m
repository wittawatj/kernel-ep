classdef Kernel < handle
    %KERNEL Abstract class for kernels
    
    properties
    end
    
    methods (Abstract)
        % Evaluate this kernel on the data1 and data2 where both are
        % are Instances.
        Kmat = eval(this, data1, data2);
        
        % Evaluate k(x1, y1), k(x2, y2), .... where X and Y are Instances.
        % X and Y must have the same number of instances.
        Kvec = pairEval(this, X, Y);
    end
    
end

