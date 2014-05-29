classdef GenericMVMapper2In < DistMapper2
    %GENERICMVMAPPER2IN A distribution mapper taking 2 Distribution's.
    % and outputs another Distribution constructed from the specified
    % DistBuilder.
    %   - Use incomplete Cholesky internally. Call CondCholFiniteOut.
    %   - Use an InstancesMapper with kernel supporting TensorInstances of 2
    %   MV1Instances. 
    
    properties (SetAccess=private)
        % a conditional mean embedding operator
        operator;
        % DistBuilder used for constructing the right output Distribution.
        distBuilder;
    end
    
    methods
        function this=GenericMVMapper2In(operator, distBuilder)
            assert(isa(operator, 'InstancesMapper'));
            assert(isa(distBuilder, 'DistBuilder'));
            
            this.operator = operator;
            this.distBuilder = distBuilder;
        end
        
        
        function dout = mapDist2(this, din1, din2)
            assert(isa(din1, 'Distribution'));
            assert(isa(din2, 'Distribution'));
            
            dins1 = MV1Instances(din1);
            dins2 = MV1Instances(din2);
            tensorIn =  TensorInstances({dins1, dins2});
            zoutStat = this.operator.mapInstances(tensorIn);
            
            builder = this.distBuilder;
            dout = builder.fromStat(zoutStat);
            assert(isa(dout, 'Distribution'), ...
                'distBuilder should construct a Distribution.');
            
        end
    end
    
      
end

