classdef GitTool < handle 
    %GIT Class containing static methods to communicate with git .
    
    properties
    end
    
    methods(Static)
        function str=getCurrentCommit()
            % return the hash of the current commit 
            % use git.m 
            try 
                % Just in case some machine does not have git command.
                % Wrap with try-catch 
                str=git('rev-parse --short=8 HEAD');
                str=strtrim(str);
            catch err 
                str = '';
            end
        end
    end
    
end

