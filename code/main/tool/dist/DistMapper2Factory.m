classdef DistMapper2Factory
    %DISTMAPPER2FACTORY A factory for creating DistMapper2 of many types.
    %
    
    properties
    end
    
    methods (Static)
        function [Map, C] = genericGauss1TensorMapper(X1Ins, X2Ins, Out, Kcandid, op)
            op.kernel_candidates = Kcandid;
            OutSuff = [[Out.mean]; [Out.variance] + [Out.mean].^2];
            
            % options for repeated hold-outs
            %             op.num_ho = 3;
            %             op.train_size = floor(0.8*n);
            %             op.test_size = min(1000, n - op.train_size);
            %             op.chol_tol = 1e-5;
            %             op.reglist = [1e-4, 1e-2, 1];
            %             op.chol_maxrank = min(500, n);
            % learn operator
            tensorIn =  TensorInstances({X1Ins, X2Ins});
            [Op, C] = CondCholFiniteOut.learn_operator(tensorIn, OutSuff,  op);
            Map = Gauss1TensorMapper2In(Op);
            
        end
        
        function [Map, C] = learnKMVMapper1D(X1, X2, Out, op)
            % - X1, X2 are array of DistNormal representing training set
            % - Out is and array of DistNormal representing the training
            % set for outputs.
            % - op = option structure
            if nargin < 4
                op=[];
            end
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
            % distance on means.
            mean_med_factors = myProcessOptions(op, 'mean_med_factors', [1/3, 1, 3]);
            assert(all(mean_med_factors>0));
            
            % Numerical array of factors to be multiplied with the median
            % distance on variances.
            variance_med_factors = myProcessOptions(op, 'variance_med_factors', ...
                [1/3, 1, 3]);
            assert(all(variance_med_factors>0));
            
            S1 = X1Ins.getAll();
            S2 = X2Ins.getAll();
            K1cell = KMVGauss1.candidates(S1, mean_med_factors, variance_med_factors,...
                med_subsamples);
            K2cell = KMVGauss1.candidates(S2, mean_med_factors, variance_med_factors,...
                med_subsamples);
            
            
            % kernel on tensor product space. Stack in the same order as in
            % TensorInstances.
            Kcandid = KProduct.cross_product(K1cell, K2cell);
            
            [Map, C] = DistMapper2Factory.genericGauss1TensorMapper(X1Ins, X2Ins, Out, Kcandid, op);
        end
        
        function [Map, C] = learnKNaturalGaussMapper1D(X1, X2, Out, op)
            % - X1, X2 are array of DistNormal representing training set
            % - Out is and array of DistNormal representing the training
            % set for outputs.
            % - op = option structure
            if nargin < 4
                op=[];
            end
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

