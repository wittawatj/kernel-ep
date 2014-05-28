classdef Params2Instances < Instances
    %PARAMS2INSTANCES Instances representing many Distribution's as
    %matrices of parameters.
    %  data variable is a struct with fields
    %  - param1 = a matrix with each column representing 1st parameter d1xn
    %  - param2 = a matrix with each column representing 2nd parameter d2xn
    %  - param1Name = name of param 1
    %  - param2Name = name of param 2
    %
    properties (SetAccess=private)
        param1;
        param2;
        param1Name;
        param2Name;
    end
    
    methods
        function this=Params2Instances(arr, p1Name, p2Name)
            if nargin < 3
                p2Name = '';
            end
            
            if nargin < 2
                p1Name = '';
            end
            % make sure it is 1d
            assert(any(size(arr)==1), 'array must be 1d');
            assert(isa(arr, 'Distribution'));
            
            C = vertcat(arr.parameters);
            this.param1 = [C{:, 1}];
            assert(size(this.param1,1)==1);
            
            this.param2 = [C{:, 2}];
            assert(size(this.param2, 1)==1);
            
            this.param1Name = p1Name;
            this.param2Name = p2Name;
        end
        
        function Data=get(this, Ind)
            Data = struct('param1', this.param1(:, Ind), ...
                'param2', this.param2(:, Ind), ...
                'param1Name', this.param1Name, ...
                'param2Name', this.param2Name);
        end
        
        function Data=getAll(this)
            Data = struct('param1', this.param1, ...
                'param2', this.param2, ...
                'param1Name', this.param1Name, ...
                'param2Name', this.param2Name);
        end
        
        
        function Ins=instances(this, Ind)
            
            % make a dummy Params2Instances
            Ins = MV1Instances(DistNormal(0, 1), this.param1Name, ...
                this.param2Name);
            Ins.param1 = this.param1(:, Ind);
            Ins.param2 = this.param2(:, Ind);
        end
        
        function l = count(this)
            l = length(this.param1);
        end
    end
end

