classdef DefaultMsgBundle < MsgBundle
    %DEFAULTMSGBUNDLE Default implementation of MsgBundle
    %    .
    
    properties (SetAccess=protected)
        % a cell array C of DistArray. C{i} gives the DistArray for variable i.
        inDistArrays;

        % output DistArray
        outDistArray;
    end

    properties
        % string description
        description;
    end
    
    methods
        function this=DefaultMsgBundle(outDistArray, varargin)
            % varargin{i} is DistArray for variable i
            assert(isa(outDistArray, 'DistArray'));
            assert(~isempty(varargin), 'at least one input is needed.');
            C = varargin;
            for i=1:length(C)
                assert(isa(C{i}, 'DistArray'));
            end
            % check count()
            f = @(da)(da.count());
            Counts=cellfun(f, C);
            assert(length(unique(Counts))==1);

            this.inDistArrays = C;
            this.outDistArray = outDistArray;
        end

        % index = 1,..getDegree(). Return a DistArray for the input dimenion 
        % specified by the index.
        function distArray=getInputBundle(this, index)
            assert(index <= this.getDegree() && index>=1);
            distArray=this.inDistArrays{index};
        end

        function daCell=getInputbundles(this)
            daCell=this.inDistArrays;

        end

        % return the number of incoming variables (connected variables to a factor)
        function d=numInVars(this)
            d=length(this.inDistArrays);
        end

        % a DistArray representing array of output messages.
        function distArray=getOutBundle(this)
            distArray=this.outDistArray;
        end

        % return a bundle of incoming messages given the instanceIndex.
        function inMsgs=getInputMsgs(instanceIndex)
            nv = this.numInVars();
            C=cell(1, nv); 
            for i=1:nv
                % da is a DistArray
                da = this.inDistArrays{i};
                C{i} = da.get(instanceIndex);
            end 
            inMsgs=C;

        end

        function [trBundle, teBundle]=splitTrainTest(this, trProportion)
            % trBundle, teBundle are DefaultMsgBundle.
            assert(trProportion>0 && trProportion<1, ... 
                'trProportion must be in (0, 1).');
            nv=this.numInVars();
            n=this.count();
            trCount=max(1, floor(trProportion*n));

            trInDists=cell(1, nv);
            teInDists=cell(1, nv);
            
            trI=randperm(n, trCount);
            % make indices for test samples
            teI=false(1, n);
            teI(trI)=true;
            teI=find(~teI);
            for i=1:nv
                % da is a DistArray for one variable
                da=this.inDistArrays{i};
                trInDists{i}=da.instances(trI);
                assert(isa(trInDists{i}, 'DistArray'));
                teInDists{i}=da.instances(teI);
                assert(isa(teInDists{i}, 'DistArray'));
            end
            trOut=this.outDistArray.instances(trI);
            assert(isa(trOut, 'DistArray'));
            teOut=this.outDistArray.instances(teI);
            assert(isa(teOut, 'DistArray'));
            trBundle=DefaultMsgBundle(trOut, trInDists{:});
            teBundle=DefaultMsgBundle(teOut, teInDists{:});

        end

        function msgBundle=subsample(this, n)
            assert(n>0, 'subsample size must be positive');
            if n>=this.count()
                msgBundle=this;
                return;
            end

            nv=this.numInVars();
            newInDas=cell(1, nv);
            I=randperm(this.count(), n);
            for i=1:nv
                da=this.inDistArrays{i};
                newInDas{i}=da.instances(I);
            end
            newOutDa=this.outDistArray.instances(I);
            msgBundle=DefaultMsgBundle(newOutDa, newInDas{:});

        end

        % The number of instance pairs.
        function n = count(this)
            % check count()
            f = @(da)(da.count());
            Counts=cellfun(f, this.inDistArrays);
            assert(length(unique(Counts))==1);

            da = this.inDistArrays{1};
            n = da.count();
        end

        function s=getDescription(this)
            s=this.description;
        end

        %%%%%%%%%%%%%
        function s = saveobj(this)
            s.inDistArrays=this.inDistArrays;
            s.outDistArray=this.outDistArray;
        end
    end % end methods
    
    methods (Static)
        function obj = loadobj(s)
            % loadobj must be Static so it can be called without object
            indas=s.inDistArrays;
            obj=DefaultMsgBundle(s.outDistArray, indas{:});
        end
    end

end

