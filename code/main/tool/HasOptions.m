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

    end
    
end

