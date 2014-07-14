classdef GenericMapper < DistMapper2 & DistMapper
    %GENERICMAPPER2IN A distribution mapper taking 2 Distribution's.
    % and outputs statistics used for constructing 
    % another Distribution using the specified DistBuilder.
    %   - Use an InstancesMapper supporting TensorInstances of 2 DistArray's
    %    
    
    properties (SetAccess=private)
        % a conditional mean embedding operator
        operator;
        % DistBuilder used for constructing the right output Distribution.
        distBuilder;

        %number of incoming variables
        numInVars;
    end
    
    methods
        function this=GenericMapper(operator, distBuilder, numInVars)
            assert(isa(operator, 'InstancesMapper'));
            assert(isa(distBuilder, 'DistBuilder'));
            assert(numInVars > 0);
            
            this.operator = operator;
            this.distBuilder = distBuilder;
            this.numInVars=numInVars;
        end
        
        
        function dout = mapDist2(this, din1, din2)
            assert(isa(din1, 'Distribution'));
            assert(isa(din2, 'Distribution'));
            
            dout=this.mapDists(din1, din2);
        end

        %%%%%%%%% Methods from DistMapper %%%%%

        % Produce an output distribution dout from input distributions 
        % (instances of Distribution)
        function dout=mapDists(this, varargin)
            assert(length(varargin)==this.numInVars(), ...
                'expect %d incoming messages.', this.numInVars());
            C=varargin;
            in=cell(1,length(C));
            for i=1:length(C)
                in{i}=DistArray(C{i});
            end
            tensorIn = TensorInstances(in);
            % Assume the operator outputs a matrix containing distribution 
            % statistics.
            zoutStat = this.operator.mapInstances(tensorIn);
            
            builder = this.distBuilder;
            dout = builder.fromStat(zoutStat);
            assert(isa(dout, 'Distribution'), ...
                'distBuilder should construct a Distribution.');

        end

        function douts=mapDistArrays(this, varargin)
            % lazy implementation
            % Return a DistArray of output messages.
            nv=length(varargin);
            assert(nv==this.numInVars(), ...
                'expect %d incoming messages.', this.numInVars());
            C=varargin;
            flength=@(da)da.count();
            L=cellfun(flength, C);
            % check that all DistArray's have the same length
            assert(length(unique(L))==1);
            n=unique(L);
            douts=cell(1, n);
            for i=1:n
                in=cell(1, nv);
                for j=1:nv
                    da=C{j};
                    assert(isa(da, 'DistArray'));
                    in{j}=da.get(i);
                    assert(isa(in{j}, 'Distribution'));
                end
                douts{i}=this.mapDists(in{:});
            end
            % convert to DistArray
            arrOfDists=[douts{:}];
            douts=DistArray(arrOfDists);
        end

        % Same as mapDists() with input messages represented by an instance 
        % of InMsgs
        function dout=mapInMsgs(this, inMsgs)
            assert(isa(inMsgs, 'InMsgs'));
            C=inMsgs.getMsgs();
            dout=this.mapDists(C{:});

        end
        % return the number of incoming messages
        % this mapper expects to take.
        function nv=numInVars(this)
            nv=this.numInVars;
        end


        function s = shortSummary(this)
            s = sprintf('%s(%s, %s)', mfilename, this.operator.shortSummary(),...
                this.distBuilder.shortSummary);
        end
    end
    
      
end

