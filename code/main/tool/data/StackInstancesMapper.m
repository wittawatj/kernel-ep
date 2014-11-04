classdef StackInstancesMapper < InstancesMapper
    %STACKINSTANCESMAPPER An InstancesMapper which stacks other InstancesMapper's
    %   - The output of each InstancesMapper is assumed to be a vector.
    
    properties
        % cell array of InstancesMapper's
        instancesMappers;
    end
    
    methods
        function this=StackInstancesMapper(varargin)
            % varargin = cell array of InstancesMapper's. Order does matter. 
            % The stacking is done in order.
            %
            assert(all(cellfun(@(x)(isa(x, 'InstancesMapper')), varargin)))
            assert(~isempty(varargin));
            this.instancesMappers = varargin;

        end
        % Map Instances Xin into Zout. The type of Zout is not restricted.
        function Zout = mapInstances(this, Xin)
            l = length(this.instancesMappers);
            Zcell = cell(1, l);
            for i=1:l
                map = this.instancesMappers{i};
                Zcell{i} = map.mapInstances(Xin);
            end
            Zout = vertcat(Zcell{:});
        end

        % return a short summary of this mapper
        function s = shortSummary(this)
            mappers = this.instancesMappers;
            st = sprintf('%s(%s', mfilename, mappers{1}.shortSummary());
            for i=2:length(mappers)
                mi = mappers{i};
                st = sprintf('%s, %s', st, mi.shortSummary());
            end
            s = [st, ')'];
        end
    end %end methods
    
end

