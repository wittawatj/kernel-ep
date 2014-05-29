classdef DistMapper2Factory
    %DISTMAPPER2FACTORY A factory for creating DistMapper2 of many types.
    %
    
    properties
    end
    
    methods (Static)
        function [Map, C] = genericGauss1TensorMapper(X1Ins, X2Ins, Out, Kcandid, op)
            op.kernel_candidates = Kcandid;
            OutSuff = [[Out.mean]; [Out.variance] + [Out.mean].^2];
            
            % learn operator
            tensorIn =  TensorInstances({X1Ins, X2Ins});
            [Op, C] = CondCholFiniteOut.learn_operator(tensorIn, OutSuff,  op);
            Map = Gauss1TensorMapper2In(Op);
            
        end
        
        
        function [Map, C] = learnKMVMapper1D(msgBundle, op)
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
            X1Ins = MV1Instances(X1);
            X2Ins = MV1Instances(X2);
            
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
            
            % The DistBuilder for the output distribution. For example, if
            % Out is an array of DistBeta, it makes sense to use
            % DistBetaBuilder. However in general one can use
            % DistNormalBuilder which will constructs normal distribution
            % messages output.
            % Default to the same type as Out array.
            outBuilder = myProcessOptions(op, 'out_msg_distbuilder', Out(1).getDistBuilder() );
            
            S1 = X1Ins.getAll();
            S2 = X2Ins.getAll();
            K1cell = KMVGauss1.candidates(S1, mean_med_factors, variance_med_factors,...
                med_subsamples);
            K2cell = KMVGauss1.candidates(S2, mean_med_factors, variance_med_factors,...
                med_subsamples);
            
            % kernel on tensor product space. Stack in the same order as in
            % TensorInstances.
            Kcandid = KProduct.cross_product(K1cell, K2cell);
            op.kernel_candidates = Kcandid;
            
            % learn operator
            tensorIn =  TensorInstances({X1Ins, X2Ins});
            outStat = outBuilder.getStat(Out);
            [Op, C] = CondCholFiniteOut.learn_operator(tensorIn, outStat,  op);
            Map = GenericMVMapper2In(Op, outBuilder);
            
        end
        
        function [Map, C] = learnKParamsMapper1D(msgBundle, op)
            % - X1, X2 are array of Distribution representing training set.
            % - Out is and array of DistNormal representing the training
            % set for outputs.
            % - op = option structure
            % Learn a mapper using product kernel of KParams2Gauss1.
            % For DistNormal X1, X2, this is equivalent to KMVGauss1
            % because parameters are {mean, variance}.
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
            assert(isa(Out, 'DistNormal'));
            assert(length(X1)==length(X2));
            n = length(X1);
            X1Ins = Params2Instances(X1);
            X2Ins = Params2Instances(X2);
            
            % number of samples to be used for computing the pairwise
            % median distance. The pairwise median distance is just a
            % heuristic. We do not need a precise median. So subsampling
            % suffices.
            med_subsamples = myProcessOptions(op, 'med_subsamples', min(1500, n));
            
            % Numerical array of factors to be multiplied with the median
            % distance on param1.
            param1_med_factors = myProcessOptions(op, 'param1_med_factors', [1/3, 1, 3]);
            assert(all(param1_med_factors>0));
            
            % Numerical array of factors to be multiplied with the median
            % distance on param2.
            param2_med_factors = myProcessOptions(op, 'param2_med_factors', ...
                [1/3, 1, 3]);
            assert(all(param2_med_factors>0));
            
            S1 = X1Ins.getAll();
            S2 = X2Ins.getAll();
            K1cell = KParams2Gauss1.candidates(S1, param1_med_factors, param2_med_factors,...
                med_subsamples);
            K2cell = KParams2Gauss1.candidates(S2, param1_med_factors, param2_med_factors,...
                med_subsamples);
            
            % kernel on tensor product space. Stack in the same order as in
            % TensorInstances.
            Kcandid = KProduct.cross_product(K1cell, K2cell);
            op.kernel_candidates = Kcandid;
            OutSuff = [[Out.mean]; [Out.variance] + [Out.mean].^2];
            
            % learn operator
            tensorIn =  TensorInstances({X1Ins, X2Ins});
            [Op, C] = CondCholFiniteOut.learn_operator(tensorIn, OutSuff,  op);
            Map = ParamsTensorMapper2In(Op);
            
        end
        
        
        function [Map, C] = learnKNaturalGaussMapper1D(msgBundle, op)
            % - X1, X2 are array of DistNormal representing training set
            % - Out is and array of DistNormal representing the training
            % set for outputs.
            % - op = option structure
            if nargin < 2
                op=[];
            end
             
            assert(isa(msgBundle, 'MsgBundle2'));
            X1 = msgBundle.getLeftMessages();
            X2 = msgBundle.getRightMessages();
            Out = msgBundle.getOutMessages();
            
            assert(isa(X1, 'DistNormal'));
            assert(isa(X2, 'DistNormal'));
            assert(isa(Out, 'DistNormal'));
            assert(length(X1)==length(X2));
            n = length(X1);
            X1Ins = MV1Instances(X1);
            X2Ins = MV1Instances(X2);
            
            % number of samples to be used for computing the pairwise
            % median distance. The pairwise median distance is just a
            % heuristic. We do not need a precise median. So subsampling
            % suffices.
            med_subsamples = myProcessOptions(op, 'med_subsamples', min(1500, n));
            
            % Numerical array of factors to be multiplied with the median
            % distance on precision*mean.
            pm_med_factors = myProcessOptions(op, 'prec_mean_med_factors', [1/3, 1, 3]);
            assert(all(pm_med_factors>0));
            
            % Numerical array of factors to be multiplied with the median
            % distance on negative precision
            np_med_factors = myProcessOptions(op, 'neg_prec_med_factors', ...
                [1/3, 1, 3]);
            assert(all(np_med_factors>0));
            
            S1 = X1Ins.getAll();
            S2 = X2Ins.getAll();
            K1cell=KNaturalGauss1.candidates(S1, pm_med_factors, np_med_factors, med_subsamples);
            K2cell=KNaturalGauss1.candidates(S2, pm_med_factors, np_med_factors, med_subsamples);
            
            % kernel on tensor product space. Stack in the same order as in
            % TensorInstances.
            Kcandid = KProduct.cross_product(K1cell, K2cell);
            
            [Map, C] = DistMapper2Factory.genericGauss1TensorMapper(X1Ins, X2Ins, Out, Kcandid, op);
        end
    end
    
end

