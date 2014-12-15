classdef HerdInstancesMapper < InstancesMapper
    % HERDINSTANCESMAPPER An InstancesMapper based on herding weights 
    %   -.
    %    
    properties(SetAccess = protected)
        % weight vector in Lasso
        weights;

        % intercept term. A real number.
        intercept;

        % sufficient statistic of the target variable to send to 
        targetSuff;

        % output location points. Output refers to z as in f(z|x1, x2,...)
        outLocation;

        % conditioning location points
        condPoints;

    end
    
    methods
        function this=HerdInstancesMapper(weights, intercept, targetSuff, ...
                outLoc, cond_points)

            % We can keep just non-zero weights here. 
            %
            assert(isnumeric(weights));
            assert(isnumeric(intercept));
            assert(isnumeric(outLoc));
            assert(isa(cond_points, 'MatTensorInstances'));
            assert(length(cond_points) == size(outLoc, 2));
            assert(size(targetSuff, 1)==1, 'targetSuff from Lasso must be 1-dim');
            assert(size(outLoc, 2) == length(targetSuff));
            assert(isnumeric(targetSuff));

            nzI = abs(weights) > 1e-9;
            %nzI = weights ~= 0;
            this.weights = weights(nzI);
            this.intercept = intercept;
            % targetSuff is just one sufficient statistic e.g., mean. 
            % There may be other kinds of sufficient statistic needed e.g., 
            % uncentered second moment. 
            this.targetSuff = targetSuff(nzI);
            this.outLocation = outLoc(:, nzI);
            this.condPoints = cond_points.instances(nzI);
        end

        function Zout = mapInstances(this, Xin)
            % Map TensorInstances (of DistArray's) Xin into expected 
            % sufficient statistic
            condPointsCell = this.condPoints.matsCell;
            inDaCell = Xin.instancesCell;
            outDa = inDaCell{1};
            outLoc = this.outLocation;
            targetSuff = this.targetSuff;
            assert(size(targetSuff, 1) == 1);
            intercept = this.intercept;

            K = length(this.condPoints);
            n = length(Xin);
            Zout = zeros(1, n);
            for j=1:n
                % 1 x n
                outDaj = outDa(j);
                Den = outDaj.density(outLoc);
                for i=2:length(inDaCell)
                    celli = inDaCell{i};
                    mij = celli(j);
                    condPointsi = condPointsCell{i-1};
                    mijDen = mij.density(condPointsi);
                    Den = Den.*mijDen;
                end
                Zout(j) = targetSuff(:)'*Den(:) + intercept;
            end
        end

        % return a short summary of this mapper
        function s = shortSummary(this)
            s = sprintf('%s(K=%d, icept=%.3g)', mfilename, length(this.weights), ...
                this.intercept);
        end

    end %end methods
    
end

