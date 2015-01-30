classdef DefaultFactorOperator < FactorOperator
    %DEFAULTFACTOROPERATOR Default implementation of FactorOperator
    %
    
    properties(SetAccess=private)
        % cell array of DistMapper's in order of indices to be used 
        % with getDistMapper(index).
        distMappers;

        % string for shortSummary()
        summary;
    end
    
    methods
        function this=DefaultFactorOperator(distMappers, summary)
            assert(iscell(distMappers));
            assert(~isempty(distMappers), 'distMappers cannot be empty');
            % Check: all are DistMapper
            Log=cellfun(@(dm)isa(dm, 'DistMapper'), distMappers);
            assert(all(Log));
            assert(ischar(summary));

            this.distMappers=distMappers;
            this.summary=summary;
        end

        function dm=getDistMapper(this, index)
            assert(index>=1 && index<=length(this.distMappers), ...
                'index out of range');
            dm=this.distMappers{index};
            assert(isa(dm, 'DistMapper'));
        end

        % return the number of incoming messages
        % this mapper expects to take.
        function nv=numInVars(this)
            nv=length(this.distMappers);
            assert(nv>0);
        end

        function s=shortSummary(this)
            s=this.summary;
        end

        % From PrimitiveSerializable interface.
        function s=toStruct(this)
            s.className=class(this);
            C=cell(1, length(this.distMappers));
            for i=1:length(C)
                dm=this.distMappers{i};
                % DistMapper implements PrimitiveSerializable
                C{i}=dm.toStruct();
            end
            s.distMappers=C;
            s.summary=this.summary;
        end
    end
    
end

