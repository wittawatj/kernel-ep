classdef DistBetaArray < DistArray
    %DISTBETAARRAY DistArray of DistBeta's
    %   - The main reason for this extension of DistArray is for efficient 
    %   density()  method.
    %   - Compatible with DistArray
    
    properties (SetAccess=protected)
        %The following properties are inherited.
        %
        %distArray;
        %mean;
        %variance;
        %parameters;
        %d;
    end
    
    methods
        function this = DistBetaArray(distArr)
            % distArr is expected to be an array of DistBeta or DistArray
            if ~( ~isa(distArr, 'DistArray') || isa(distArr.distArray, 'DistBeta') )
                % If distArr is a DistArray that does not contain DistBeta 
                error('Input DistArray must be an array of DistBeta s');
            end
            this@DistArray(distArr);

        end

        function D=density(this, X)
            assert(isnumeric(X));
            % naive implementation for now.
            [d, n] = size(X);
            D = zeros(n, length(this.distArray)); 
            for i=1:length(this.distArray)
                dist = this.distArray(i);
                den = dist.density(X);
                D(:, i) = den(:);
            end
        end

    end % end methods
    
end

