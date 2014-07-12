classdef FactorBundles < handle
    %FACTORBUNDLES A set of MsgBundle's from a factor, one for each outgoing 
    %direction. 
    %    Incoming MsgBundle's are shared.
    
    properties(SetAccess=protected)
        % cell arrays of incoming messages
        inDistArrays;
        % cell arrays of outgoing messages (DistArray)
        outDistArrays;
    end
    
    methods
        function this=FactorBundles(inDistArrays, outDistArrays)
            % expect both inDistArrays and outDistArrays to be cell arrays of 
            % DistArray's
            assert(iscell(inDistArrays));
            assert(iscell(outDistArrays));
            % same number of variables
            assert(length(inDistArrays)==length(outDistArrays));
            daf=@(o)(isa(o, 'DistArray'));
            assert(all(cellfun(daf, inDistArrays)));
            assert(all(cellfun(daf, outDistArrays)));

            % check length
            countf=@(da)da.count();
            assert(length(unique(cellfun(countf, inDistArrays)))==1);
            assert(length(unique(cellfun(countf, outDistArrays)))==1);

            this.inDistArrays=inDistArrays;
            this.outDistArrays=outDistArrays;
            
        end

        function nv=numVars(this)
            % return total number of variables connected to the factor
            nv=length(this.inDistArrays);

        end

        function bundle=getMsgBundle(this, varIndex)
            % varIndex is from 1,..numVars()
            outda=this.outDistArrays{varIndex};
            indas=this.inDistArrays;
            bundle=DefaultMsgBundle(outda, indas{:});

        end

        function s=saveobj(this)
            s.inDistArrays=this.inDistArrays;
            s.outDistArrays=this.outDistArrays;
        end

    end
    
    methods(Static)
        function obj=loadobj(s)
            obj=FactorBundles(s.inDistArrays, s.outDistArrays);
        end
    end
end

