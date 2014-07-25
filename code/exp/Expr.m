classdef Expr < handle
    %EXPR Class mainly containing static convenient methods for experiment scripts.
    %    .
    
    properties
    end
    
    methods(Static)
        function fpath=expSavedFile(expNum, fname)
            % construct a full path the the file name fname in the experiment 
            % folder identified by expNum
            expNumFolder=Expr.expSavedFolder(expNum);
            fpath=fullfile(expNumFolder, fname);
        end

        function expNumFolder=expSavedFolder(expNum)
            % return full path to folder used for saving results of experiment 
            % identified by expNum
            assert(isscalar(expNum));
            assert(mod(expNum, 1)==0);
            root=Global.getSavedFolder();
            expFolder=fullfile(root, 'exp');
            if ~exist(expFolder, 'dir')
                mkdir(expFolder);
            end

            fname=sprintf('exp%d', expNum);
            expNumFolder=fullfile(expFolder, fname);
            if ~exist(expNumFolder, 'dir')
                mkdir(expNumFolder);
            end


        end

    end
    
end

