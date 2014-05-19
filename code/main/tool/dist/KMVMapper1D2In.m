classdef KMVMapper1D2In < DistMapper2
    %KMVMAPPER1D2IN A distribution mapper taking 2 one-dimensional DistNormal's
    % and outputs another DistNormal.
    %   - Use incomplete Cholesky internally. Call CondCholFiniteOut.
    %   - Default kernel = product of two KMVGauss1's
    
    properties
        % a conditional mean embedding operator
        operator;
    end
    
    methods
        function this=KMVMapper1D2In(operator)
            assert(isa(operator, 'InstancesMapper'));
            this.operator = operator;
        end
        
        
        function dout = mapDist2(this, din1, din2)
            assert(isa(din1, 'DistNormal'));
            assert(isa(din2, 'DistNormal'));
            
            dins1 = Gauss1Instances(din1);
            dins2 = Gauss1Instances(din2);
            tensorIn =  TensorInstances({dins1, dins2});
            zout = this.operator.mapInstances(tensorIn);
            
            % we are working with a 1d Gaussian (for now)
            % mean = zout(1), uncenter 2nd moment = zout(2)
            dout = DistNormal(zout(1), zout(2)-zout(1)^2);
            
        end
    end
    
    methods (Static)
        
        function [Map, C] = learnMapper(X1, X2, Out, op)
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
            X1Ins = Gauss1Instances(X1);
            X2Ins = Gauss1Instances(X2);
            
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
            
            tensorIn =  TensorInstances({X1Ins, X2Ins});
            % kernel on tensor product space. Stack in the same order as in
            % TensorInstances.
            Kcandid = KProduct.cross_product(K1cell, K2cell);
            
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
            [Op, C] = CondCholFiniteOut.learn_operator(tensorIn, OutSuff,  op);
            Map = KMVMapper1D2In(Op);
        end
    end
    
end

