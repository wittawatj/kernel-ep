classdef OptionsDescription < handle
    %OPTIONSDESCRIPTION Object specifying a list of possible options.
    %    .
    
    properties(SetAccess=protected)
        % a struct
        keyValueStruct;
    end
    
    methods
        function this=OptionsDescription(kv)
            % kv is a struct with representing key-value pairs
            assert(isstruct(kv));
            this.keyValueStruct=kv;
        end

        function show(this)
            % longest key length
            L=length(OptionsDescription.longestKey(this.keyValueStruct));
            L=max(L, 8);
            F=fieldnames(this.keyValueStruct);
            st=this.keyValueStruct;
            format=['%', num2str(L), 's -> %s'];
            display(sprintf(format, 'options', 'description'));
            for i=1:length(F)
                k=F{i};
                display(sprintf(format, k, st.(F{i})));
            end
        end

    end

    methods (Static)
        function k=longestKey(st)
            F=fieldnames(st);
            k='';
            for i=1:length(F)
                if length(k)<=length(F{i})
                    k=F{i};
                end

            end

        end
    end %end static methods
    
end

