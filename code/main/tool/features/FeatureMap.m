classdef FeatureMap < handle & PrimitiveSerializable
    %FEATUREMAP A finite-dimensional features generator.
    %This is useful for implementing primal solutions using random features, for instance. 

    properties
    end
   
    methods (Abstract)
        % Generate a feature vector Z given some input 
        Z=genFeatures(this, in);  

        % Generate feature vectors in the form of DynamicMatrix.
        % This is useful when in contains so many instances that storing feature 
        % vectors in the usual numerical matrix requires too much memory.
        M=genFeaturesDynamic(this, in);

        % Return a generator (function handle) to be used with DynamicMatrix
        g=getGenerator(this, in);


        % Return a new FeatureMap such that the internal weight parameters are 
        % kept and the number of features is extended. This method only makes 
        % sense to a FeatureMap which allows the control of number of features
        % such as random feature map.
        fm=cloneParams(this, numFeatures);

        % Return the number of features to be generated.
        D=getNumFeatures(this);

        % Short summary of this FeatureMap. Useful if in the form
        % mapName(param1, param2).
        s=shortSummary(this);
    end
    
end

