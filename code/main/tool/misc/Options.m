classdef Options < handle
    %OPTIONS A set of key-value pairs to be used to specify options.
    %    .
    
    properties(SetAccess=protected)
        % struct storing options
        opStruct;
    end
    
    methods
        function this=Options()
            this.opStruct=struct();
        end

        function addOptions(this, options)
            % options can be a struct or Options
            % If a key exists, its value will be replaced.
            assert(isa(options, 'Options') || isstruct(options));
            if isstruct(options)
                F=fieldnames(options);
                for i=1:length(F)
                    this.setOption(F{i}, options.(F{i}) );
                end
            else
                % an Options. Recursion.
                this.addOptions(options.opStruct);
            end

        end

        function setOption(this, key, value)
            assert(ischar(key));
            s=this.opStruct;
            % raise an error if key contains space
            s.(key)=value;
        end

    end
    
end

