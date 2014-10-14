classdef Global < handle
    %GLOBAL Class containing methods to deal with global structure of kernel-ep
    %    .
    
    properties
    end
    
    methods(Static)
        function r=getRootFolder()
            [p,f,e]=fileparts(which('startup.m'));
            r=p;
        end

        function f=getScriptFolder()
            % return the folder containing directly runnable scripts
            p=Global.getRootFolder();
            f=fullfile(p, 'script');
            if ~exist(f, 'dir')
                mkdir(f);
            end
        end

        function f=getSavedFolder()
            % return the top folder for saving .mat files 

            p=Global.getRootFolder();
            f=fullfile(p, 'saved');
            if ~exist(f, 'dir')
                mkdir(f);
            end
        end
    end
    
end

