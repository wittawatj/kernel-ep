classdef GenericMapper2In < DistMapper2
    %GENERICMAPPER2IN A distribution mapper taking 2 Distribution's.
    % and outputs statistics used for constructing 
    % another Distribution using the specified DistBuilder.
    %   - Use an InstancesMapper supporting TensorInstances of 2 DistArray's
    %    
    
    properties (SetAccess=private)
        % a conditional mean embedding operator
        operator;
        % DistBuilder used for constructing the right output Distribution.
        distBuilder;
    end
    
    methods
        function this=GenericMapper2In(operator, distBuilder)
            assert(isa(operator, 'InstancesMapper'));
            assert(isa(distBuilder, 'DistBuilder'));
            
            this.operator = operator;
            this.distBuilder = distBuilder;
        end
        
        
        function dout = mapDist2(this, din1, din2)
            assert(isa(din1, 'Distribution'));
            assert(isa(din2, 'Distribution'));
            
            %dins1 = MV1Instances(din1);
            %dins2 = MV1Instances(din2);
            dins1 = DistArray(din1);
            dins2 = DistArray(din2);
            tensorIn =  TensorInstances({dins1, dins2});
            % Assume the operator outputs a matrix containing distribution 
            % statistics.
            zoutStat = this.operator.mapInstances(tensorIn);
            
            builder = this.distBuilder;
            dout = builder.fromStat(zoutStat);
            assert(isa(dout, 'Distribution'), ...
                'distBuilder should construct a Distribution.');
            
        end

        function s = shortSummary(this)
            s = sprintf('%s(%s, %s)', mfilename, this.operator.shortSummary(),...
                this.distBuilder.shortSummary);
        end
    end
    
      
end

