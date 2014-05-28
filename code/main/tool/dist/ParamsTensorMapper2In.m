classdef ParamsTensorMapper2In < DistMapper2
    %PARAMSTENSORMAPPER2IN A distribution mapper taking 2 Distribution's. 
    % and outputs another DistNormal.
    %   - Use incomplete Cholesky internally. Call CondCholFiniteOut.
    %   - Use an InstancesMapper with kernel supporting TensorInstances of 2
    %   Params2Instances 
    
    properties (SetAccess=private)
        % a conditional mean embedding operator
        operator;
    end
    
    methods
        function this=ParamsTensorMapper2In(operator)
            assert(isa(operator, 'InstancesMapper'));
            this.operator = operator;
        end
        
        
        function dout = mapDist2(this, din1, din2)
            assert(isa(din1, 'Distribution'));
            assert(isa(din2, 'Distribution'));
            
            dins1 = Params2Instances(din1);
            dins2 = Params2Instances(din2);
            tensorIn =  TensorInstances({dins1, dins2});
            zout = this.operator.mapInstances(tensorIn);
            
            % we are working with a 1d Gaussian (for now)
            % mean = zout(1), uncenter 2nd moment = zout(2)
            dout = DistNormal(zout(1), zout(2)-zout(1)^2);
            
        end
    end
    
    
end

