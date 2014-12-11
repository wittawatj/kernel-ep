classdef SigmoidCondFactor < CondFactor
    %SIGMOIDCONDFACTOR Sigmoid factor 
    %   - f(z|x) where x is one dimension.
    
    properties (SetAccess = protected)
    end
    
    methods
        function this = SigmoidCondFactor()
        end


        % return the number of incoming variables (connected variables to a factor)
        function d=numInVars(this)
            d = 2;
        end

        % sample from this factor conditioning on cond_samples. 
        % cond_samples must be an instance of MatTensorInstances.
        function z = sample(this, cond_samples)
            assert(isa(cond_samples, 'MatTensorInstances') );
            matsCell = cond_samples.matsCell;
            assert(length(matsCell) == 1);
            x = matsCell{1};
            x = x(:)';
            z = 1./(1+exp(-x));
        end

        function matTenIns = batchDrawQuasiMCPoints(this, K)
            %Xloc = linspace(-20, 20, K);
            Xloc = linspace(-16, 16, K);
            cond_matTenIns = MatTensorInstances({Xloc});
            %Zloc = this.sample(cond_matTenIns);
            %matTenIns = MatTensorInstances({Zloc, Xloc});
            matTenIns = cond_matTenIns;
        end

        % return a short summary of this factor
        function s = shortSummary(this)
            s = sprintf('%s', mfilename );
        end

    end
    
end

