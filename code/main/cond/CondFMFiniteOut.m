classdef CondFMFiniteOut < InstancesMapper & PrimitiveSerializable
    %CONDFMFINITEOUT Generic operator using 
    %finite-dimensional feature map (primal solution) i.e., k(x,y)=phi(x)*phi(y)
    %where phi(x) is explicit and finite-dimensional.
    %
    properties (SetAccess=private)
        
        featureMap;
        %regParam; %regularization parameter

        % matrix needed in mapInstances(). dz x numFeatures
        % where dz = dimension of output sufficient statistic.
        mapMatrix;
    end

    methods (Static)
        function Ax=ax(x, dm, lambda)
            % Return (P*P'+lambda*eye(D))x without forming P*P'
            assert(isa(dm, 'DynamicMatrix'));
            %D=size(dm, 1);
            %Ptx=dm.t().rmult(x);
            Ptx=dm.lmult(x')';
            Ax=dm.rmult(Ptx)+lambda*x;

        end
    end

    methods

        function this = CondFMFiniteOut(fm, In, Out, lambda, use_multicore)
            % fm = a FeatureMap
            % In = training Instances 
            % Out = a matrix of outputs
            % lambda = regularization parameter
            %
            if nargin < 5
                use_multicore = true;
            end
            assert(isnumeric(Out));
            assert(isa(In, 'Instances'));
            assert(isa(fm, 'FeatureMap'));
            assert(isnumeric(lambda) && lambda >= 0, ...
                'regularization param must be non-negative.');

            %this.Out = Out;
            this.featureMap = fm;
            %this.regParam = lambda;

            %% It is not a good idea to explicitly form P (Dxn)
            %P = fm.genFeatures(In); % D x n where D = numFeatures
            %D = size(P, 1);
            %Q = Out*P'; % dy x D
            %% This step can be extremely expensive. O(D^3). 
            %% But need to be done only once.
            %A = P*P' + lambda*eye(D);

            % Do the lines above with a DynamicMatrix
            % dm should be treated as P (Dxn) but dynamically generated.
            dm = fm.genFeaturesDynamic(In);
            D = size(dm, 1);
            Q = dm.rmult(Out')'; %dyxD
            % PPT is DxD where D is the number of primal features.
            % D > 1e4 and it may take up too much memory.
            if ~use_multicore
                PPT = dm.mmt();
                clear dm
                A = PPT + lambda*eye(D);

                opts.POSDEF = true;
                opts.SYM = true;
                T = linsolve(A', Q', opts)';
            else 
                F=Q'; %Dxdy
                % Forming PPT (DxD) can take up too much memory and computation.
                % Try conjugate gradient.
                % Solve for T in AT=F
                afunc=@(x)CondFMFiniteOut.ax(x, dm, lambda);
                tol=1e-5;
                maxit=30;
                use_multicore=true;

                if use_multicore
                    gop=globalOptions();
                    multicore_settings.multicoreDir= gop.multicoreDir;                    
                    multicore_settings.maxEvalTimeSingle=60*100;
                    colSolve=@(col)pcg(afunc, col, tol, maxit)';
                    % Gather columns
                    cols=cell(1, size(F, 2));
                    for i=1:length(cols)
                        cols{i}=F(:, i);
                    end
                    resultCell = startmulticoremaster(colSolve, cols, multicore_settings);
                    T=vertcat(resultCell{:});
                    assert(~isempty(T), 'empty map matrix');
                    assert(all(size(T)==size(Q)));
                else
                    dout=size(Out,1);
                    T=zeros(dout, D);
                    % not use multicore 
                    for i=1:dout
                        fi=F(:, i);
                        T(i, :)=pcg(afunc, fi, tol, maxit)';
                    end
                end
            end

            this.mapMatrix = T;

        end


        function Zout = mapInstances(this, Xin)
            % Map Instances in Xin to Zout with this operator.
            assert(isa(Xin, 'Instances'));
            T = this.mapMatrix;
            fm = this.featureMap;
            Pin = fm.genFeatures(Xin);

            try 
                Zout = T*Pin;
                %display('size of T: ')
                %display(size(T));
                %display('size of Pin: ')
                %display(size(Pin));
            catch err 
                display(err);
            end
        end

        function s = shortSummary(this)
            s = sprintf('%s(%s)', mfilename, this.featureMap.shortSummary());
        end

        % From PrimitiveSerializable interface 
        function s=toStruct(this)
            s.className=class(this);
            s.featureMap=this.featureMap.toStruct();
            %s.regParam=this.regParam;
            s.mapMatrix=this.mapMatrix;
        end

        %function s=saveobj(this)
        %s.featureMap=this.featureMap;
        %s.regParam=this.regParam;
        %s.mapMatrix=this.mapMatrix;
        %end

    end %end methods

    methods (Static)

        %function obj=loadobj(s)
        %% This values are just to make the constructor happy.
        %fakeIn=DistArray(DistNormal(0, 1));
        %fakeOut=[1,2]';
        %obj= CondFMFiniteOut(s.featureMap, fakeIn, fakeOut, s.regParam);
        %assert(isa(s.featureMap, 'FeatureMap'));
        %obj.featureMap=s.featureMap;
        %assert(isnumeric(s.mapMatrix));
        %obj.mapMatrix=s.mapMatrix;

        %end

        function [Op, C] = learn_operator(In, Out,  op)
            % In is likely to be a TensorInstances
            assert(isa(In, 'Instances'));
            [ C] = cond_fm_finiteout( In, Out, op );
            bestMap = C.bfeaturemap;
            lambda = C.blambda;
            % Allow the change of numFeatures in case the number is reduced 
            % during  LOOCV.
            op.num_primal_features = myProcessOptions(op, 'num_primal_features', ...
                2e3);
            map = bestMap.cloneParams(op.num_primal_features);
            Op = CondFMFiniteOut(map, In, Out, lambda);
        end

        function [Op, C] = learn_operator_cmaes(In, Out,  op)
            % Learn the operator with black-box optimization CMA-ES
            % In is likely to be a TensorInstances
            assert(isa(In, 'Instances'));
            [ C] = cond_fm_finiteout_cmaes( In, Out, op );
            bestMap = C.bfeaturemap;
            lambda = C.blambda;
            % Allow the change of numFeatures in case the number is reduced 
            % during  LOOCV.
            op.num_primal_features = myProcessOptions(op, 'num_primal_features', ...
                2e3);
            map = bestMap.cloneParams(op.num_primal_features);
            Op = CondFMFiniteOut(map, In, Out, lambda);
        end

    end

end

