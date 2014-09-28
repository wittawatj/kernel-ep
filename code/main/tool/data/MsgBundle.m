classdef MsgBundle < handle
    %MSGBUNDLE A immutable container for input-output message pairs for learning a DistMapper.
    %
    %
    
    properties
    end
    
    methods (Abstract)

        % index = 1,..numInVars(). Return a DistArray for the input dimension 
        % specified by the index.
        distArray=getInputBundle(this, index);

        % Return all input DistArray's in a cell array.
        daCell=getInputBundles(this);
        
        % return the number of incoming variables (connected variables to a factor)
        d=numInVars(this);

        % a DistArray representing array of output messages.
        distArray=getOutBundle(this);

        % return a bundle of incoming messages given the instanceIndex.
        inMsgs=getInputMsgs(instanceIndex);

        % Split this MsgBundle into training and testing bundles.
        % This MsgBundle should not change at all.
        % Since MsgBundle is a handle, trBundle, teBundle may refer internally 
        % to the data in this MsgBundle.
        % trProportion is in (0,1) for proportion of training samples.
        [trBundle, teBundle]=splitTrainTest(this, trProportion);

        % Partition this bundle into non-overlapping trBundle and teBundle whose 
        % sizes are given by the two arguments.
        [trBundle, teBundle]=partitionTrainTest(this, trN, teN)

        % Create a new MsgBundle by subsampling without replacement 
        % n is the number of instances. This MsgBundle should not change. 
        msgBundle=subsample(this, n);

        % The number of instance pairs.
        n = count(this);

        % return a string description
        s=getDescription(this);
        
        % short identifier for the message bundle. Useful for naming.
        s=getBundleName(this);
    end

    methods
        function n=length(this)
            n=this.count();
        end

        function T=getInputTensorInstances(this)
            % Return all input bundles in a TensorInstances.
            % Useful for training.
            T=TensorInstances(this.getInputBundles());
        end
    end

end

