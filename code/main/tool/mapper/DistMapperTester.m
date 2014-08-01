classdef DistMapperTester < handle & HasOptions
    %DISTMAPPERTESTER Generic class to test a DistMapper.
    %    What to test depends on the implementation of a subclass.
    
    properties(Abstract, SetAccess=protected)
        % The DistMapper to test
        distMapper;
    end
    
    methods(Abstract)
        % run the test implemented by a subclass on the distMapper. 
        % Return a struct containing the results.
        % testBundle is a MsgBundle. A MsgBundle will contain input-output 
        % message pairs. 
        S=testDistMapper(this, testBundle);


        % Compare the  groundTruth DistArray to the output outDa. 
        % Both DistArray's must have the same length. 
        % Typically the outDa is DistArray output from this.distMapper.
        compareOutputs(this, groundTruth, outDa);

        % summary in string of this DistMapperTester
        s=shortSummary(this);

        % Compare the output DistArray to the groundTruth DistArray.
        %compareOutputs(output, groundTruth);

    end
    
end

