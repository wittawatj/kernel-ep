classdef Options < handle
    %OPTIONS A set of key-value pairs to be used to specify options.
    %    Usage: let o=this. 
    %    o.opt('a') to get option setting of key 'a'
    %    o.opt('a', 5) set key 'a' to 5
    
    properties(SetAccess=protected )
        % struct storing options
        opStruct;
    end
    
    methods

        function this=Options(st)
            if nargin<1
                st=struct();
            end

            assert(isstruct(st));
            this.opStruct=st;
        end

        function st=toStruct(this)
            st=this.opStruct;
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

        function v=opt(this, key, value)
            if nargin < 3
                % no value is specified
                v=this.opStruct.(key);
                return;
            end
            this.setOption(key, value);
        end


        function setOption(this, key, value)
            assert(ischar(key));
            s=this.opStruct;
            % raise an error if key contains space
            s.(key)=value;
            this.opStruct=s;
        end

        function show(this)
            % longest key length
            display(this.opStruct);
        end
    end % end methods
    
end

