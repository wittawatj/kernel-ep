classdef FactorOpSerializer < handle 
    %FACTOROPSERIALIZER For serializing FactorOperator
    %   .
    
    properties(Constant)
        % Asssume relative to the first-level folder
        DEFAULT_FACTOROP_FOLDER='saved/factor_op';
        FNAME_PREFIX='factorOp_';
        SERIALIZED_FNAME_PREFIX='serialFactorOp_';
    end
    
    methods
        % load a FactorOperator by the name 
        function factorOp=loadFactorOperator(this, name)
            fpath=FactorOpSerializer.getFullPath(name);
            load(fpath);
            assert(isa(factorOperator, 'FactorOperator'));
            factorOp=factorOperator;
        end

        % save to the file path specified by (toPath)
        function saveFactorOperator(this, factorOp, name)
            % name should be unique so that it can be loaded back
            % Preferably, name should not have a space.
            assert(isa(factorOp, 'FactorOperator'));
            fpath=FactorOpSerializer.getFullPath(name);

            generateTime=clock();
            % also save current stackTrace for latter reference
            stackTrace=dbstack;
            assert(isstruct(stackTrace));
            savedFactorOp=struct();
            savedFactorOp.factorOp=factorOp;
            savedFactorOp.generateTime=generateTime;
            savedFactorOp.stackTrace=stackTrace;
            %save(fpath, 'factorOp', 'generateTime', 'stackTrace');
            save(fpath, 'savedFactorOp');
        end

        function serializeFactorOperator(this, factorOp, name)
            % Serialize the specified FactorOperator to a struct to be called 
            % from C#. 

            assert(isa(factorOp, 'FactorOperator'));
            serialFactorOp=factorOp.toStruct();
            fpath=FactorOpSerializer.getSerializeFullPath(name);

            generateTime=clock();
            % also save current stackTrace for latter reference
            stackTrace=dbstack;
            assert(isstruct(stackTrace));
            save(fpath, 'serialFactorOp', 'generateTime', 'stackTrace');
        end

    end %end methods
    
    methods(Static)
        function fpath=getFullPath(name)
            fname=sprintf('%s%s', FactorOpSerializer.FNAME_PREFIX, name);
            fpath=fullfile(FactorOpSerializer.DEFAULT_FACTOROP_FOLDER, fname);
        end

        function fpath=getSerializeFullPath(name)
            fname=sprintf('%s%s', FactorOpSerializer.SERIALIZED_FNAME_PREFIX, name);
            fpath=fullfile(FactorOpSerializer.DEFAULT_FACTOROP_FOLDER, fname);
        end
    end
end

