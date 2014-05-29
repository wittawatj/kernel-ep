classdef ExpMapper
    %EXPMAPPER Facade class containing all methods related to experiments
    %on mapper/operator.
    
    properties
    end
    
    methods (Static)
        
        function [mapper, op, C] = kgenericMapper2Test(msgBundle, msgBundleTest, op )
            % Need to specify a mapper_learner which is a function taking
            % (msgBundle, op) -> [mapper, op]
            % e.g., DistMapper2Factory.learnKMVMapper1D
            %
            % - Learn the conditional mean embedding operator on msgBundle
            % - Test the operator on a separate test set msgBumdleTest
            % - Measure the error with Hellinger distance.
            %   
            % - This function is for testing Gauss1TensorMapper2In which includes a
            % mapper using KMVGauss1 kernel and KNaturalGauss1 kernel. See
            % DistMapper2Factory for how to construct these mappers.
            %
            assert(isa(msgBundle, 'MsgBundle2'));
            
            if nargin < 3
                op = [];
            end
            assert(isfield(op, 'mapper_learner'), 'mapper_learner needed to be specified' );
            % mapper_learner: (msgBundle, op) -> [mapper, op]
            mapper_learner = op.mapper_learner;
            
            op.seed = myProcessOptions(op, 'seed', 1);
            seed = op.seed;
            oldRng = rng();
            rng(seed);
            
%             [X, T, Tout, Xte, Tte, Toutte] = Data.splitTrainTest(s, ntr, nte);
%             assert(length(X)==ntr);
%             assert(length(T)==ntr);
%             assert(length(Tout)==ntr);
            
            % This will load a bunch of variables in s into the current scope.
            % load 'n', 'op', 'a', 'w', 'X', 'T', 'Xout', 'Tout'
            % eval(structvars(100, s));
            
            % options for repeated hold-outs
            ntr = msgBundle.count();
            op.num_ho = 3;
            op.train_size = floor(0.7*ntr);
            op.test_size = min(1000, ntr - op.train_size);
            op.chol_tol = 1e-15;
            op.chol_maxrank = min(700, ntr);
            op.reglist = myProcessOptions(op, 'reglist', [1e-2, 1]);
            
            % options used in learning a mapper in DistMapper2Factory
            op.med_subsamples = min(1500, ntr);
            [mapper, C] = mapper_learner(msgBundle, op);
            
            % testing
            Xte = msgBundleTest.getLeftMessages();
            Tte = msgBundleTest.getRightMessages();
            Toutte = msgBundleTest.getOutMessages();
            [hmean, hvar, hkl]=distMapper2_tester( mapper,  Xte, Tte, Toutte);
            
            % cheating by testing on training set
            % limit = 100;
            % I = randperm(length(X), min(limit, length(X)) );
            % [hmean, hvar, hkl]=distMapper2_tester( mapper,  X(I), T(I), Tout(I));
            
            axmean = get(hmean, 'CurrentAxes');
            axvar = get(hvar, 'CurrentAxes');
            axkl = get(hkl, 'CurrentAxes');
            title(axmean, sprintf('Training size: %d', ntr));
            title(axvar, sprintf('Training size: %d', ntr));
            title(axkl, sprintf('Training size: %d', ntr));

            rng(oldRng);
        end
        
        function  [mapper, op, C] = knaturalMapper2Test(msgBundle, msgBundleTest, op )
            
            % Used when selected DistMapper2Factory.learnKNaturalGaussMapper1D
            op.prec_mean_med_factors = myProcessOptions(op, 'prec_mean_med_factors', [1]);
            op.neg_prec_med_factors = myProcessOptions(op, 'neg_prec_med_factors', [1]);
            op.mapper_learner = @DistMapper2Factory.learnKNaturalGaussMapper1D;
            [mapper, op, C] = ExpMapper.kgenericMapper2Test(msgBundle, msgBundleTest, op );
        end
        
        function  [mapper, op, C] = kmvMapper2Test(msgBundle, msgBundleTest, op )
            
            % Used when selected DistMapper2Factory.learnKMVMapper1D
            op.mean_med_factors = myProcessOptions(op, 'mean_med_factors', [1, 3]);
            op.variance_med_factors = myProcessOptions(op, 'variance_med_factors', [1, 3]);
            op.mapper_learner = @DistMapper2Factory.learnKMVMapper1D;
            [mapper, op, C] = ExpMapper.kgenericMapper2Test(msgBundle, msgBundleTest, op );
        end
        
        function [mapper, op, C] = runSigmoidKMVMapperTestRight(opnew)
            if nargin < 1
                opnew = [];
            end
            sigmoid_bundle_path = myProcessOptions(opnew, 'sigmoid_bundle_path', ...
                SigmoidFactor.DEFAULT_BUNDLE_PATH);
            load(sigmoid_bundle_path);
            % op will be loaded
            op = dealstruct(op, opnew);
            % Expect: 
%             save(sigmoid_bundle_path,  'op',  'msgBundle2Left', ...
%                 'msgBundle2Right', 'generateMethod', 'generateTime');
            
            op.mapper_train_size = myProcessOptions(op, 'mapper_train_size', 2000);
            ntr = op.mapper_train_size ;
            op.mapper_test_size = myProcessOptions(op, 'mapper_test_size', 100);
            nte = op.mapper_test_size;
            trBundle = msgBundle2Right.removeRandom(ntr);
            msgBundle2Right.reduceTo(nte);
            teBundle = msgBundle2Right;
            
            [mapper, op, C] = ExpMapper.kmvMapper2Test(trBundle, teBundle, op );
        end
        
        
        function [mapper, op, C] = runSigmoidKMVMapperTestLeft(opnew)
            if nargin < 1
                opnew = [];
            end
            sigmoid_bundle_path = myProcessOptions(opnew, 'sigmoid_bundle_path', ...
                SigmoidFactor.DEFAULT_BUNDLE_PATH);
            load(sigmoid_bundle_path);
            % op will be loaded
            op = dealstruct(op, opnew);
            % Expect: 
%             save(sigmoid_bundle_path,  'op',  'msgBundle2Left', ...
%                 'msgBundle2Right', 'generateMethod', 'generateTime');
            
            op.mapper_train_size = myProcessOptions(op, 'mapper_train_size', 2000);
            ntr = op.mapper_train_size ;
            op.mapper_test_size = myProcessOptions(op, 'mapper_test_size', 100);
            nte = op.mapper_test_size;
            trBundle = msgBundle2Left.removeRandom(ntr);
            msgBundle2Left.reduceTo(nte);
            teBundle = msgBundle2Left;
            
            [mapper, op, C] = ExpMapper.kmvMapper2Test(trBundle, teBundle, op );
        end
    end
    
end

