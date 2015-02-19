classdef BayesLinRegFM < UAwareInstancesMapper & PrimitiveSerializable
    %BAYESLINREGFM Bayesian linear regression using 
    %finite-dimensional feature map (primal solution) i.e., k(x,y)=phi(x)*phi(y)
    %where phi(x) is explicit and finite-dimensional.
    % - one-dimensional output (regression target)
    %
    properties (SetAccess=private)
        
        featureMap;
        %regParam; %regularization parameter

        % matrix needed in mapInstances(). dz x numFeatures
        % where dz = dimension of output sufficient statistic.
        mapMatrix;

        % posterior covariance matrix. Used for computing predictive variance.
        % DxD where D = number of features
        posteriorCov;

        % output noise variance (regularization parameter)
        noise_var;

        % cross correlation between input and output.
        % Useful for online learning update.
        % = XY'.
        crossCorrelation;
    end

    methods

        function this = BayesLinRegFM(fm, In, Out, noise_var )
            % fm = a FeatureMap
            % In = training Instances 
            % Out = a row vector of outputs
            % noise_var = output noise variance. This is proportional to the 
            %   regularization parameter in linear regression.
            %
            assert(isnumeric(Out));
            assert(size(Out, 1) == 1);
            assert(isa(In, 'Instances'));
            assert(isa(fm, 'FeatureMap'));
            assert(isnumeric(noise_var) && noise_var >= 0, ...
                'regularization param must be non-negative.');

            %this.Out = Out;
            Y = Out;
            this.featureMap = fm;
            %this.regParam = noise_var;

            X = fm.genFeatures(In); % D x n where D = numFeatures
            D = size(X, 1);
            postPrec = X*X' + noise_var*eye(D);
            % This step can be extremely expensive. O(D^3). 
            % But need to be done only once.
            postCov = inv(postPrec);

            % We can use \. But since we will need to store inverse any way.
            this.crossCorrelation = X*Y(:);
            this.mapMatrix = ( postCov*this.crossCorrelation )';
            this.posteriorCov = postCov;
            this.noise_var = noise_var;
        end

        function U = estimateUncertainty(this, Xin)
            % U: 1xn vector of predictive variances
            assert(isa(Xin, 'Instances'));

            fm = this.featureMap;
            Pin = fm.genFeatures(Xin);
            Sigma = this.posteriorCov;
            SP = Sigma*Pin;
            U = sum(SP.*Pin, 1) + this.noise_var;
            assert(all(size(U) == [1, length(Xin)]));

        end

        function Zout  = mapInstances(this, Xin)
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


        function [Zout, U] = mapInstancesAndU(this, Xin)
            assert(isa(Xin, 'Instances'));
            T = this.mapMatrix;
            fm = this.featureMap;
            Pin = fm.genFeatures(Xin);
            Zout = T*Pin;

            Sigma = this.posteriorCov;
            SP = Sigma*Pin;
            U = sum(SP.*Pin, 1) + this.noise_var;
            assert(all(size(U) == [1, length(Xin)]));
        end

        function s = shortSummary(this)
            s = sprintf('%s(%s)', mfilename, this.featureMap.shortSummary());
        end

        % From PrimitiveSerializable interface 
        function s=toStruct(this)
            s = struct();
            s.className=class(this);
            s.featureMap=this.featureMap.toStruct();
            %s.regParam=this.regParam;
            s.mapMatrix=this.mapMatrix;
            s.crossCorrelation = this.crossCorrelation;
            s.posteriorCov = this.posteriorCov; 
            s.noise_var = this.noise_var;
        end


    end %end methods

    methods (Static)

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
            if isa(bestMap, 'RFGJointKGG')
                % Gaussian kernel on mean embeddings uses two-staged approximation.
                fm = bestMap.cloneParams(op.num_primal_features, op.num_inner_primal_features);
            else 
                fm = bestMap.cloneParams(op.num_primal_features);
            end
            Op = BayesLinRegFM(fm, In, Out, lambda );
        end

    end

end

