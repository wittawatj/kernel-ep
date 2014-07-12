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

        function newOd=merge(this, od)
            % merge OptionsDescription od into this.
            % If there are duplicate keys, keys (and values) 
            % in od are kept.
            assert(isa(od, 'OptionsDescription'));
            newStruct=dealstruct(this.keyValueStruct, od);
            newOd=OptionsDescription(newStruct);
        end

        function show(this)
            display(this.keyValueStruct);
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

