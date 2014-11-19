classdef MPDictionary < handle
    %MPDICTIONARY Dictionary i.e., set of MPFunctionClass to be used as a container 
    %for all candidate functions in MatchingPursuit.
    %   .
    
    properties(SetAccess = protected)
        % cell array of MPFunctionClass's
        functionClasses;

    end
    
    methods
        function this = MPDictionary(fClasses)
            assert(iscell(fClasses));
            assert(all(cellfun(@(f)(isa(f, 'MPFunctionClass')), fClasses )));
            this.functionClasses = fClasses;
        end

        function G = evaluate(this, X)
            % Evaluate all marked functions on the samples
            % X is an instance of Instances.
            % return: G (#marked x sample size)

        end

        function b = getBasisCount(this)
        end

    end
    
end

