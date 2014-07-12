classdef BundleGenerator < handle
    %BUNDLEGENERATOR A MsgBundle generator. 
    %   A dataset (incoming-outgoing message pairs) generator for a specific 
    %   problem should extend this class.
    
    properties
    end
    
    methods (Abstract)
        % generate a MsgBundle which can be used to train an operator
        % n is the number of message pairs to generate.
        % varOutIndex specifies the output variable to which messages will be 
        % sent to. This affects bundle.getOutBundle().
        bundle=genBundle(this, n, varOutIndex);

        % Generate a MsgBundle for each outgoing direction. 
        % All MsgBundle's are in a FactorBundles fbundles
        fbundles=genBundles(this, n);


        % number of total connected variables to this factor
        nv=numVars(this);

        %s is a string describing this generator.
        s=shortSummary(this);

    end

    methods 
        % generate a MsgBundle and serialize to a file specified by the name
        % Refer to BundleSerializer for how to load back
        function genBundleTo(this, name)
            bundle=this.genBundle();
            writer=BundleSerializer();
            writer.saveBundle(bundle, name);
        end

    end
    
end

