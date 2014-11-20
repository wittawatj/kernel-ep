classdef HasOptions < handle
    %HASOPTIONS Interface annotating that an object has Options
    %    .
    
    properties (Abstract, GetAccess=public, SetAccess=protected )
        % An instance of Options
        % Subclass should initially set this field to default options.
        options;
    end
    
    methods (Abstract)
        % Return an instance of OptionsDescription describing possible options.
        od=getOptionsDescription(this);

        % return an instance of Options specify default options.
        op=getDefaultOptions(this);

    end

    methods 
        % for convenience  
        % So, we can do this.opt(..) instead of this.options.opt(...)
        function v=opt(this, key, value)
            assert(isa(this.options, 'Options'));
            if nargin < 3
                v=this.options.opt(key);
                return;
            end
            this.options.setOption(key, value);
        end

        % Add all options in options. Can be instance of Options or struct.
        function addOptions(this, options)
            this.options.addOptions(options);
        end

        function has=hasKey(this, key)
            has = this.options.hasKey(key);
        end

        function v=isNoKeyOrEmpty(this, key)
            % return true if there is no specified option key or the value is 
            % empty.
            assert(ischar(key));
            v = ~this.hasKey(key) || isempty(this.opt(key));

        end

    end
    
end

