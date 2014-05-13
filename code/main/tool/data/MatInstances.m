classdef MatInstances < Instances
    %MATINSTANCES Data matrix where each column represents one instance.
    
    
    properties (SetAccess=private)
        % the actual matrix
        Mat;
    end
    
    methods
        function this=MatInstances(X)
            this.Mat = X;
        end
        
        function Data=get(this, Ind)
            Data = this.Mat(:, Ind);
        end
        
        
        function Data=getAll(this)
            % Return all data instances.
            Data = this.get(1:this.count());
        end
        
        
        function Ins=instances(this, Ind)
            Ins = MatInstances(this.Mat(:, Ind));
        end
        
        function l = count(this)
            % total number of instances 
            l = size(this.Mat, 2);
        end
           
        function M=matrix(this)
            M=this.Mat;
        end
        
    end
    
end

