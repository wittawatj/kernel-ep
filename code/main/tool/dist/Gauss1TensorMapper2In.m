classdef Gauss1TensorMapper2In < DistMapper2
    %GAUSS1TENSORMAPPER2IN A distribution mapper taking 2 one-dimensional 
    % DistNormal's and outputs another DistNormal.
    %   - Use incomplete Cholesky internally. Call CondCholFiniteOut.
    %   - Use an InstancesMapper with kernel supporting TensorInstances of 2
    %   Gauss1Instances. 
    
    properties (SetAccess=private)
        % a conditional mean embedding operator
        operator;
    end
    
    methods
        function this=Gauss1TensorMapper2In(operator)
            assert(isa(operator, 'InstancesMapper'));
            this.operator = operator;
        end
        
        
        function dout = mapDist2(this, din1, din2)
            assert(isa(din1, 'DistNormal'));
            assert(isa(din2, 'DistNormal'));
            
            dins1 = Gauss1Instances(din1);
            dins2 = Gauss1Instances(din2);
            tensorIn =  TensorInstances({dins1, dins2});
            zout = this.operator.mapInstances(tensorIn);
            
            % we are working with a 1d Gaussian (for now)
            % mean = zout(1), uncenter 2nd moment = zout(2)
            dout = DistNormal(zout(1), zout(2)-zout(1)^2);
            
        end
    end
    
    
end

