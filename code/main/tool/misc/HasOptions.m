classdef HasOptions < handle
    %HASOPTIONS Interface annotating that an object has Options
    %    .
    
    properties (Abstract, GetAccess=public )
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
    
end

