classdef DistMapperSerializer < handle
    %DISTMAPPERSERIALIZER Object used to serialize DistMapper to a file
    %   .
    
    properties(Constant)
        % Asssume relative to the first-level folder
        DEFAULT_DISTMAPPER_FOLDER='saved/distmapper';
        FNAME_PREFIX='distMapper_';

    end
    
    methods
        % load a DistMapper by the name 
        function dm=loadDistMapper(this, name)
            fpath=DistMapperSerializer.getFullPath(name);
            % expect an object named 'distMapper'
            load(fpath);
            assert(isa(distMapper, 'DistMapper'));
            dm=distMapper;
        end

        % save to the file path specified by (toPath)
        function saveDistMapper(this, dm, name)
            % also save with name 'distMapper'
            % name should be unique so that it can be loaded back
            % Preferably, name should not have a space.
            assert(isa(dm, 'DistMapper'));
            fpath=DistMapperSerializer.getFullPath(name);

            generateTime=clock();
            % also save current stackTrace for latter reference
            stackTrace=dbstack;
            assert(isstruct(stackTrace));
            distMapper=dm;
            save(fpath, 'distMapper', 'generateTime', 'stackTrace');
        end

        function ex=exist(this, name)
            % Return non-zero if a DistMapper specified by the name exists on 
            % the file system.

            fpath=DistMapperSerializer.getFullPath(name);
            ex=exist([fpath, '.mat'], 'file');

        end

    end %end methods
    
    methods(Static)
        function fpath=getFullPath(name)
            fname=sprintf('%s%s', DistMapperSerializer.FNAME_PREFIX, name);
            fpath=fullfile(DistMapperSerializer.DEFAULT_DISTMAPPER_FOLDER, fname);

        end
    end

end

