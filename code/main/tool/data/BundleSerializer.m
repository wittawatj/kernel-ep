classdef BundleSerializer < handle
    %BUNDLESERIALIZER Object used to serialize (load/save) a MsgBundle to a file.
    %   .
    
    properties(Constant)
        % Asssume relative to the first-level folder
        DEFAULT_BUNDLE_FOLDER='saved/bundle';
        FNAME_PREFIX='msgBundle_';

    end
    
    methods

        % load a MsgBundle by the name 
        function bundle=loadBundle(this, name)
            fpath=BundleSerializer.getFullPath(name);
            % expect an object named 'msgBundle'
            load(fpath);
            assert(isa(msgBundle, 'MsgBundle'));
            bundle=msgBundle;
        end

        % save to the file path specified by (toPath)
        function saveBundle(this, msgBundle, name)
            % also save with name 'msgBundle'
            % name should be unique so that it can be loaded back
            % Preferably, name should not have a space.
            assert(isa(msgBundle, 'MsgBundle'));
            fpath=BundleSerializer.getFullPath(name);

            generateTime=clock();
            % also save current stackTrace for latter reference
            stackTrace=dbstack;
            assert(isstruct(stackTrace));
            save(fpath, 'msgBundle', 'generateTime', 'stackTrace');
        end

    end %end methods
    
    methods(Static)
        function fpath=getFullPath(name)
            fname=sprintf('%s%s', BundleSerializer.FNAME_PREFIX, name);
            fpath=fullfile(BundleSerializer.DEFAULT_BUNDLE_FOLDER, fname);

        end
    end

end

