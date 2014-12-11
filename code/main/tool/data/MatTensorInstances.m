classdef MatTensorInstances < Instances
    %MATTENSORINSTANCES Tensor instances of matrices.  
    %   .

    properties (SetAccess=private)
        % cell array of matrices
        matsCell;
        
    end
    
    methods
        function this=MatTensorInstances(matsCell)
            assert(iscell(matsCell));
            assert(~isempty(matsCell));
            % make sure all Instances have the same count
            countf = @(in)(size(in, 2));
            counts = cellfun(countf, matsCell);
            assert(length(unique(counts))==1);
        
            this.matsCell = matsCell;
        end
        
        function Data=get(this, Ind)
            % Return data as a cell array of length = length(matsCell)
            mats = this.matsCell;
            f = @(in)(in(:, Ind));
            Data = cellfun(f, mats, 'UniformOutput', false);
        end
        
        function Data=getAll(this)
            Data = this.matsCell;
        end
        

        function Ins=instances(this, Ind)
            Data = this.get(Ind);
            Ins = MatTensorInstances(Data);
        end
        
        function l = count(this)
            % All matrices should have the same count
            mats = this.matsCell;
            l = size(mats{1}, 2);
        end

        function dim = tensorDim(this)
            % Return the size of tensor product.
            dim = length(this.matsCell);
        end

    end %end methods
end

