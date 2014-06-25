classdef DistMapper2PrimalFactory
    %DISTMAPPER2PRIMALFACTORY A factory for creating DistMapper2 using 
    %primal features. See DistMapper2Factory for variations.
    %Remember "2" in the name means 2 input variables.
    %
    
    properties
    end
    
    methods (Static)
        
        function [Map, C] = learnRFGMVMapper(msgBundle, op)
            % Learn RandFourierGaussMVMap mapper 
            % - op = option structure
            if nargin < 2
                op=[];
            end
            assert(isa(msgBundle, 'MsgBundle2'));
            X1 = msgBundle.getLeftMessages();
            X2 = msgBundle.getRightMessages();
            
            % Return an array of ground truth output messages.
            Out = msgBundle.getOutMessages();
            
            assert(isa(X1, 'Distribution'));
            assert(isa(X2, 'Distribution'));
            assert(isa(Out, 'Distribution'));
            assert(length(X1)==length(X2));
            assert(length(X2)==length(Out));
            n = length(X1);
            % DistArray will not nest. The constructor prevents it.
            X1Ins = DistArray(X1);
            X2Ins = DistArray(X2);
            
            % number of samples to be used for computing the pairwise
            % median distance. The pairwise median distance is just a
            % heuristic. We do not need a precise median. So subsampling
            % suffices.
            med_subsamples = myProcessOptions(op, 'med_subsamples', min(1500, n));
            
            % Numerical array of factors to be multiplied with the median
            % distance on means.
            mean_med_factors = myProcessOptions(op, 'mean_med_factors', [1/3, 1, 3]);
            assert(all(mean_med_factors>0));
            
            % Numerical array of factors to be multiplied with the median
            % distance on variances.
            variance_med_factors = myProcessOptions(op, 'variance_med_factors', ...
                [1/3, 1, 3]);
            assert(all(variance_med_factors>0));
            
            % number of primal features to use for candidates. 
            % This number is not necessarily the same during the test time.
            % Typically during LOOCV, the number of features can be low to make
            % it fast to select a candidate.
            candidate_primal_features = myProcessOptions(op, 'candidate_primal_features', 3000);

            % The DistBuilder for the output distribution. For example, if
            % Out is an array of DistBeta, it makes sense to use
            % DistBetaBuilder. However in general one can use
            % DistNormalBuilder which will constructs normal distribution
            % messages output.
            % Default to the same type as Out array.
            outBuilder = myProcessOptions(op, 'out_msg_distbuilder', Out(1).getDistBuilder() );
            
            tensorIn =  TensorInstances({X1Ins, X2Ins});
            FMcell = RandFourierGaussMVMap.candidates(tensorIn, mean_med_factors, variance_med_factors,...
                candidate_primal_features, med_subsamples);
            
            op.featuremap_candidates = FMcell;
            
            % learn operator
            outStat = outBuilder.getStat(Out);
            [Op, C] = CondFMFiniteOut.learn_operator(tensorIn, outStat,  op);
            Map = GenericMapper2In(Op, outBuilder);
            
        end
        
    end % end static methods
    
end

