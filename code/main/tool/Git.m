classdef Git < handle 
    %GIT Class containing static methods to communicate with git .
    
    properties
    end
    
    methods(Static)
        function str=getCurrentCommit()
            % return the hash of the current commit 
            % use git.m 
            str=git('rev-parse --short=10 HEAD');
        end
    end
    
end

