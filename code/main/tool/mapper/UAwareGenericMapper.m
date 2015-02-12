classdef UAwareGenericMapper < GenericMapper & UAwareDistMapper
    %UAWAREGENERICMAPPER An uncertainty aware distribution mapper taking Distribution's.
    % and outputs statistics used for constructing 
    % another Distribution using the specified DistBuilder.
    %   - Use an InstancesMapper supporting TensorInstances of 2 DistArray's
    %   - Assume the InstancesMapper outputs statistics to be used with 
    %   distBuilder.fromStat()
    %    
    
    properties (SetAccess=private)
        %% an operator mapping Distribution to a vector of statistics 
        %% An InstancesMapper. Need to be PrimitiveSerializable.
        %operator;

        %% DistBuilder used for constructing the right output Distribution.
        %distBuilder;

        %%number of incoming variables
        %nv;
    end
    
    methods
        function this=UAwareGenericMapper(operator, distBuilder, numInVars)
            assert(isa(operator, 'UAwareInstancesMapper'));
            this@GenericMapper(operator, distBuilder, numInVars);
        end
        
        
        function u = estimateUncertainty(this, varargin)
            % Same interface as mapDists(this, varargin).
            op = this.operator;
            assert(isa(op, 'UAwareInstancesMapper'));
            assert(length(varargin)==this.numInVars(), ...
                'expect %d incoming messages.', this.numInVars());
            C=varargin;
            tensorIn = UAwareGenericMapper.cellDistArrayToTensor(C);
            % Assume the operator outputs a matrix containing distribution 
            % statistics.
            u = op.estimateUncertainty(tensorIn);
        end

        % output both predictions and uncertainty
        function [dout, u] = mapDistsAndU(this, varargin)

            assert(length(varargin)==this.numInVars(), ...
                'expect %d incoming messages.', this.numInVars());
            C=varargin;
            tensorIn = UAwareGenericMapper.cellDistArrayToTensor(C);
            % Assume the operator outputs a matrix containing distribution 
            % statistics.
            zoutStat = this.operator.mapInstances(tensorIn);
            
            builder = this.distBuilder;
            dout = builder.fromStat(zoutStat);
            assert(isa(dout, 'Distribution'), ...
                'distBuilder should construct a Distribution.');
            u = this.operator.estimateUncertainty(tensorIn);
        end

        function U = estimateUDistArrays(this, varargin)
            C = varargin;
            %tensorIn = UAwareGenericMapper.cellDistArrayToTensor(C);
            U = this.estimateUncertainty(C{:});
        end
    end

    methods (Static)
        function T = cellDistArrayToTensor(C)
            assert(iscell(C));
            in=cell(1,length(C));
            for i=1:length(C)
                assert(isa(C{i}, 'Distribution'));
                in{i}=DistArray(C{i});
            end
            T = TensorInstances(in);
        end

    end
    
      
end

