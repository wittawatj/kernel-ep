classdef DistMapperLearner < handle & HasOptions
    %DISTMAPPERLEARNER A class used to learn a DistMapper from training data.
    %    .
    
    properties
    end
    
    methods (Abstract)
        % learn a DistMapper given the training data in MsgBundle.
        dm=learnDistMapper(this);

        s=shortSummary(this);


    end
    
end

