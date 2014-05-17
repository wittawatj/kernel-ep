classdef Gauss1Instances < Instances
    %GAUSS1INSTANCES Instances representing many 1d Gaussian distribution as
    %matrices of mean's and variance's
     %  data variable is a struct with fields
    %  - mean = a matrix with each column representing one mean. dxn
    %  - variance = If d==1, a row vector. If d>1, a dxdxn matrix.
    %  - d = dimension of the distribution (= size(mean,1))
    %
    properties (SetAccess=private)
        mean;
        variance;
    end
    
    methods
        function this=Gauss1Instances(arr)
            % make sure it is 1d
            assert(any(size(arr)==1), 'array must be 1d');
            assert(isa(arr, 'DistNormal'));
            assert(arr(1).d==1);
            this.mean = [arr.mean];
            assert(size(this.mean, 1)==1);
            this.variance = [arr.variance];
            assert(size(this.variance, 1)==1);
        end
        
        function Data=get(this, Ind)
            Data = struct('d', 1, 'mean', this.mean(Ind), 'variance', ...
                this.variance(Ind));
        end
        
        function Data=getAll(this)
            Data = struct('d', 1, 'mean', this.mean, 'variance', ...
                this.variance);
        end
        

        function Ins=instances(this, Ind)
            arr = DistNormal(this.mean(Ind), this.variance(Ind));
            Ins= Gauss1Instances(arr);
        end
        
        function l = count(this)
            l = length(this.mean);
        end
    end
end

