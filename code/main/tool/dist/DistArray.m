classdef DistArray < Distribution & Instances
    %DISTARRAY Distribution array. 
    %  Subclass of Distribution so that it has the same interface. 
    %  Also an Instances.
    %  The main reason to consider DistArray is to cache mean, variance properties.
    %  Given D = array of DistNormal's, doing [D.mean] is very slow.
    %  It is expected that each Distribution in DistArray does not change because 
    %   DistArray will cache some properties. 
    %

    properties (SetAccess=protected)
        distArray;
        % cached means 
        mean;
        % cached variance
        variance;
        % cached parameters. This is a cell array of parameters (which is also
        % a cell array).
        parameters;

    end
    
    methods

        function this = DistArray(distArr)
            % distArr is expected to be an array of Distribution
            assert(isa(distArr, 'Distribution'));
            % If distArr is itself a DistArray, make sure not to nest two 
            % DistArray's inside one another.
            if isa(distArr, 'DistArray')
                this.distArray = distArr.distArray;
            else
                this.distArray = distArr;
            end

        end

        function M = get.mean(this)
            if isempty(this.mean) 
                this.mean = [this.distArray.mean];
            end
            M = this.mean;
        end

        function V = get.variance(this)
            if isempty(this.variance)
                nu = numel(this.distArray(1).variance);
                if nu > 1
                    % multivariate Gaussian
                    this.variance = cat(3, this.distArray.variance);
                else
                    % univariate Gaussian. 
                    this.variance = [this.distArray.variance];
                end
            end
            V = this.variance; 
        end

        function D=density(this, X)
            error('density() evaluated on a DistArray is not supported yet.');

        end

        function P = get.parameters(this)
            if isempty(this.parameters)
                this.parameters = { this.distArray.parameters };
            end
            % P will be a cell array
            P = this.parameters;
        end

        function p=isProper(this)
            % return a logical array
            p = arrayfun(@(x)(x.isProper()), this.distArray);
        end

        function names=getParamNames(this)
            names=this.distArray(1).getParamNames();
        end
        %%%%%%%%%%%%%%%%
        %%% For Instances interface
        
        function Data=get(this, Ind)
            Data = this.distArray(Ind);
        end
        
        function Data=getAll(this)
            Data = this.distArray;
        end
        
        function Ins=instances(this, Ind)
            subd = this.distArray(Ind);
            Ins = DistArray(subd);

        end
        
        function l = count(this)
            l = length(this.distArray);
        end

        %%%%%%%%%%%%
        function s=saveobj(this)
            % an array of Distribution's
            s.distArray=this.distArray;
        end

    end

    methods(Static)
        function obj=loadobj(s)
            obj=DistArray(s.distArray);
        end
    end
    
end

