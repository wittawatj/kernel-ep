classdef ExpPrimalMapper
    %EXPPRIMALMAPPER Facade class containing all methods related to experiments
    %on mapper/operator using primal features.
    
    properties

    end
    
    methods (Static)
        
        
        
        function [mapper, op, Helling, C] = runSigmoidMapperTestRight(opnew)
            % Test sending to t in p(x|t). In SigmoidFactor t is Gaussian, x is Beta.
            if nargin < 1
                opnew = [];
            end
            sigmoid_bundle_path = myProcessOptions(opnew, 'sigmoid_bundle_path', ...
                SigmoidFactor.DEFAULT_BUNDLE_PATH);
            load(sigmoid_bundle_path);
            % op will be loaded
            % Respect opnew. Deal it to the loaded op.
            op = dealstruct(op, opnew);
            % Expect: 
%             save(sigmoid_bundle_path,  'op',  'msgBundle2Left', ...
%                 'msgBundle2Right', 'generateMethod', 'generateTime');
            
            op.mapper_train_size = myProcessOptions(op, 'mapper_train_size', 2000);
            ntr = op.mapper_train_size;
            op.mapper_test_size = myProcessOptions(op, 'mapper_test_size', 100);
            nte = op.mapper_test_size;
            trBundle = msgBundle2Right.removeRandom(ntr);
            msgBundle2Right.reduceTo(nte);
            teBundle = msgBundle2Right;
            
            op.mean_med_factors = myProcessOptions(op, 'mean_med_factors', [1, 3]);
            op.variance_med_factors = myProcessOptions(op, 'variance_med_factors', [1, 3]);
            op.mapper_learner = myProcessOptions(op, 'mapper_learner', ...
                @DistMapper2PrimalFactory.learnRFGMVMapper);

            % number of primal features to use during the test time.
            % This is not the same as candidate_primal_features.
            % Actually used in CondFMFiniteOut.learn_operator
            op.num_primal_features = myProcessOptions(op, ...
                'num_primal_features', 5000);

            % number of primal features to use for candidates. 
            % Typically during LOOCV, the number of features can be low to make
            % it fast to select a candidate.
            op.candidate_primal_features = myProcessOptions(op, ...
                'candidate_primal_features', 2000);

            [mapper, op, Helling, C] = ExpMapper.kgenericMapper2Test(trBundle, teBundle, op );
        end
        
        
        function [mapper, op, Helling, C] = runSigmoidMapperTestLeft(opnew)
            % Testing sending to x in p(x|t).In SigmoidFactor t is Gaussian, x is Beta.
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
            
            op.mean_med_factors = myProcessOptions(op, 'mean_med_factors', [1, 3]);
            op.variance_med_factors = myProcessOptions(op, 'variance_med_factors', [1, 3]);
            op.mapper_learner = myProcessOptions(op, 'mapper_learner', ...
                @DistMapper2PrimalFactory.learnRFGMVMapper);

            % number of primal features to use
            op.num_primal_features = myProcessOptions(op, 'num_primal_features', 3000);

            [mapper, op, Helling, C] = ExpMapper.kgenericMapper2Test(trBundle, teBundle, op );
        end

    end %end static
    
end

