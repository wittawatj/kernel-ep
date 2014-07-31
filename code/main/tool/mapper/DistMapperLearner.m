classdef DistMapperLearner < handle & HasOptions
    %DISTMAPPERLEARNER A class used to learn a DistMapper from training data.
    %    .
    
    properties
    end
    
    methods (Abstract)
        % learn a DistMapper given the training data in MsgBundle.
        % Log is some side information generated during the learning of the 
        % DistMapper dm. Log is a struct.
        [ dm, Log ]=learnDistMapper(this);

        s=shortSummary(this);


    end
    
end

