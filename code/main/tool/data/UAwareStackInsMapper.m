classdef UAwareStackInsMapper < UAwareInstancesMapper & StackInstancesMapper
    %UAWARESTACKINSMAPPER An InstancesMapper which stacks other UAwareInstancesMapper's
    %   - The output of each UAwareInstancesMapper is assumed to be a vector.
    
    properties
        % cell array of InstancesMapper's. Defined in StackInstancesMapper
        %instancesMappers;
    end
    
    methods
        function this=UAwareStackInsMapper(varargin)
            % varargin = cell array of UAwareInstancesMapper's. Order does matter. 
            % The stacking is done in order.
            %
            assert(all(cellfun(@(x)(isa(x, 'UAwareInstancesMapper')), varargin)))
            this@StackInstancesMapper(varargin{:});

        end

        function U = estimateUncertainty(this, Xin)
            % u: mxn vector of uncertainties where n = length(instances) and m
            % is length(instancesMappers);
            assert(isa(Xin, 'Instances'));
            m = length(this.instancesMappers);
            U = zeros(m, length(Xin));
            for i=1:m
                map = this.instancesMappers{i};
                U(i, :) = map.estimateUncertainty(Xin);
            end

        end

    end %end methods
    
end

