classdef ArrayInstances < Instances
    %ARRAYINSTANCES Instances internally represented as an array.
    %   Each element in the array is one instance. Can be an object array.
    
    
    properties (SetAccess=private)
        % an array as a row vector
        arr;
        
    end
    
    methods
        function this=ArrayInstances(arr)
            % make sure it is 1d
            assert(any(size(arr)==1), 'array must be 1d');
            % store as a row
            this.arr = arr(:)';
        end
        
        function Data=get(this, Ind)
            Data = this.arr(Ind);
        end
        
        function Data=getAll(this)
%             Data = this.get(1:this.count());
            Data = this.arr;
        end
        

        function Ins=instances(this, Ind)
            Ins=ArrayInstances(this.get(Ind));
        end
        
        function l = count(this)
            l = length(this.arr);
        end
    end
    
end

