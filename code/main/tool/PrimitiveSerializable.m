classdef PrimitiveSerializable < handle 
    %PRIMITIVESERIALIZABLE Interface annotating that an object can be serialized
    %into nested structs so that it can be read back by other systems e.g., C#.
    
    properties
    end
    
    methods (Abstract)
        % convert the object to a struct s.
        % This is a bit different from saveobj(.) as the struct must not contain 
        % any objects. Only primitive type values are allowed. 
        % This implies that if a object is PrimitiveSerializable, all of 
        % its composited  objects must also be.
        %
        % Cell arrays are allowed. 
        % Necessary: s.className = 'SomeClassName'
        % 
        s=toStruct(this);

    end
    
end

