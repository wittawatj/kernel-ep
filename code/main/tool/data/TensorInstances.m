classdef TensorInstances < Instances
    %TENSORINSTANCES Meta Instances allowing a combination of many
    %Instances's
    %  The main purpose of this is to form a joint Instances in a tensor
    %  product space. This is intended to be used with KProduct.
    
    properties (SetAccess=private)
        % cell array of Instances's
        instancesCell;
        
    end
    
    methods
        function this=TensorInstances(Ins)
            % Ins = cell array of many Instances's
            assert(iscell(Ins));
            assert(~isempty(Ins));
            % make sure all Instances have the same count
            countf = @(in)(in.count());
            counts = cellfun(countf, Ins);
            assert(length(unique(counts))==1);
        
            this.instancesCell = Ins;
        end
        
        function Data=get(this, Ind)
            % Return data as a cell array of length = length(instancesCell)
            Ins = this.instancesCell;
            f = @(in)(in.get(Ind));
            Data = cellfun(f, Ins, 'UniformOutput', false);
        end
        
        function Data=getAll(this)
            Data = this.get(1:this.count());
        end
        

        function Ins=instances(this, Ind)
            is = this.instancesCell;
            f = @(in)(in.instances(Ind));
            C = cellfun(f, is, 'UniformOutput', false);
            Ins = TensorInstances(C);
            
        end
        
        function l = count(this)
            % All Instances should have the same count
            Ins = this.instancesCell;
            l = Ins{1}.count();
        end
    end
    
end

