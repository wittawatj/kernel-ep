classdef KLLossLearner < DistMapperLearner
    %KLLOSSLEARNER A DistMapperLearner by directly minimizing KL loss instead of 
    %squared loss on the moments.
    %   * Use gradient descent on the KL loss
    %   * The gradient depends on what is output by the InstancesMapper (e.g., 
    %   log variance, variance). To avoid complication, the DistBuilder (which determines
    %   what is output by InstancesMapper) will be chosen based on the output 
    %   in MsgBundle automatically.
    %
    
    properties(SetAccess=protected)
        % an instance of Options
        options;
    end
    
    methods
        function this=KLLossLearner()
            this.options=this.getDefaultOptions();
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';
            %kv.use_multicore=['If true, use multicore package.'];

            kv.feature_map = 'feature map used for the operator. Should accept a TensorInstances';
            kv.reg_param = 'regularization parameter';
            kv.minibatch_size = 'mini-batch size used for subsampling samples in stochastic gradient descent';
            kv.max_gd_iter = 'maximum gradient descent iterations to perform.';

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            %st.use_multicore=true;
            st.feature_map = [];
            st.reg_param = 1e-5;
            st.minibatch_size = 1e3;
            st.max_gd_iter = 100;

            Op=Options(st);
        end

        function [gm, C]=learnDistMapper(this, bundle )
            oldRng = rng();
            rng(this.opt('seed'), 'twister');

            assert(isa(bundle, 'MsgBundle'), 'input to constructor not a MsgBundle' );
            assert(bundle.count()>0, 'empty training set')

            outDa=bundle.getOutBundle();
            assert(isa(outDa, 'DistArray'));
            dout=outDa.get(1);
            assert(isa(dout, 'Distribution'));

            % learn operator
            if isa(dout, 'DistNormal') && dout.d == 1
                % output is 1d Gaussian
                [gm, C] = this.learn1DGaussian(bundle);
            else
                error('Unsupported output distribution. Example: %s', dout);
            end

            rng(oldRng);
        end

        function s=shortSummary(this)
            s=sprintf('%s', mfilename );
        end


    end

    methods (Access=private)
        function [gm, C]=learn1DGaussian(this, bundle)
            % cell array of DistArray's
            % Cannot just change to arbitrary DistBuilder because the derivative 
            % depends on what is output by the operator.
            distBuilder = DNormalLogVarBuilder();
            %distBuilder = DNormalVarBuilder();
            inputDistArrays=bundle.getInputBundles();
            tensorIn=TensorInstances(inputDistArrays);
            outDa=bundle.getOutBundle();

            % output statistic to be learned by the operator
            outStat=distBuilder.getStat(outDa);
            assert(isnumeric(outStat));

            % start gradient descent 
            max_gd_iter = this.opt('max_gd_iter');
            minibatch_size = this.opt('minibatch_size');
            feature_map = this.opt('feature_map');
            reg_param = this.opt('reg_param');
            if isempty(feature_map)
                error('feature_map options cannot be empty.');
            end
            D = feature_map.getNumFeatures();
            n = length(bundle);

            % initialize W 
            W = 10*[randn(1, D); rand(1, D)];
            flog = [];

            optOptions.Display = 'iter' ;
            optOptions.numDiff = 0; % use user-supplied gradient 
            optOptions.MaxIter = max_gd_iter;
            optOptions.progTol = 1e-5;
            optOptions.optTol = 1e-4;
            %optOptions.DerivativeCheck = 'on';
            %fobj = @(w)(this.Gauss1DObj(w, minibatch_size, feature_map, tensorIn, outStat, reg_param));
            fobj = @(w)(this.Gauss1DObjLogVarOut(w, minibatch_size, feature_map, tensorIn, outStat, reg_param));
            vecFObj = @(wvec)(this.vectorizedObj(fobj, wvec, size(W)));
            [Wstar, fval, exitflag, opt_output] = minFunc(vecFObj, W(:), optOptions);
            Wstar = reshape(Wstar, [size(outStat, 1), D]);


            %for t=1:max_gd_iter
            %    [f, grad] = Gauss1DObj(this, W, minibatch_size, feature_map, tensorIn, outStat);
            %    flog(end+1) = f;
            %    W = W 
            %end
            C.optOptions = optOptions;
            C.Wstar = Wstar;
            C.fval = fval;
            C.exitflag = exitflag;
            C.opt_output = opt_output;

            Op = LinearFMInstancesMapper(Wstar, feature_map);
            gm=GenericMapper(Op, distBuilder, bundle.numInVars());

        end

        function [f, flatg]= vectorizedObj(this, fobj, x, xshape)
            xre = reshape(x, xshape);
            [f, grad] = fobj(xre);
            flatg = grad(:);
        end

        function [f, grad] = Gauss1DObjLogVarOut(this, W, minibatch_size, feature_map, ...
                tensorIn, outStat, reg_param)
            % Used with log-variance output

            n = length(tensorIn);
            I = randperm(n, minibatch_size);
            Mu = outStat(1, I);
            LogVar = outStat(2, I);

            subIn = tensorIn.instances(I);
            Phi = feature_map.genFeatures(subIn);
            Pre = W*Phi;
            MeanDiff = Pre(1, :) - Mu;
            EPre2 = exp(Pre(2, :));
            grad1 = Phi*(MeanDiff./EPre2)'/n;
            
            MeanDiff2 = MeanDiff.^2;
            MD2Var = MeanDiff2 + exp(LogVar);
            T1Pre = -MD2Var./EPre2;
            T1 = Phi*T1Pre'/(2*n);
            T2 = mean(Phi, 2)/2;
            grad2 = T1 + T2;
            % make it the same shape as W (Dout x D)
            grad = [grad1, grad2]' + 2*reg_param*W;

            f = 0.5*mean(MD2Var./EPre2 -1 + Pre(2, :) - LogVar) + reg_param*W(:)'*W(:);

        end

        function [f, grad] = Gauss1DObj(this, W, minibatch_size, feature_map, ...
                tensorIn, outStat, reg_param)

            n = length(tensorIn);
            I = randperm(n, minibatch_size);
            Mu = outStat(1, I);
            Var = outStat(2, I);

            subIn = tensorIn.instances(I);
            Phi = feature_map.genFeatures(subIn);
            Pre = W*Phi;
            MeanDiff = Pre(1, :) - Mu;
            grad1 = Phi*(MeanDiff./Pre(2, :))'/n;
            
            MeanDiff2 = MeanDiff.^2;
            MD2Var = MeanDiff2 + Var;
            T1Pre = -MD2Var./(Pre(2,:).^2);
            T1 = Phi*T1Pre'/(2*n);
            T2 = Phi*(1./Pre(2, :))'/(2*n);
            grad2 = T1 + T2;
            % make it the same shape as W (Dout x D)
            grad = [grad1, grad2]' + 2*reg_param*W;
            % W may yield negative variance. 
            logOutVar = log(abs(Pre(2, :))); %!!! abs(.) is extra. Wrong !

            f = 0.5*mean(MD2Var./Pre(2, :) -1 + logOutVar - log(Var)) + reg_param*W(:)'*W(:);

        end
    end

    methods(Static)

    end
    
end

