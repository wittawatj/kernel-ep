classdef RFGJointKGGGrad2MLLearner < DistMapperLearner
    %RFGJOINTKGGGRAD2MLLEARNER A DistMapperLearner for RFGJointKGG. Parameter
    %optimization is done on the marginal likelihood in the dual form (type 2
    %ML) with gradient ascent.
    %   - The implementation forms a full gram matrix. Not suitable for large 
    %   data.
    
    properties(SetAccess=protected)
        % an instance of Options
        options;
    end

    methods
    
        function this=RFGJointKGGGrad2MLLearner()
            this.options=this.getDefaultOptions();
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';

            % The DistBuilder for the output distribution. For example, if
            % Out is an array of DistBeta, it makes sense to use
            % DistBetaBuilder. However in general one can use
            % DistNormalBuilder which will constructs normal distribution
            % messages output.
            kv.out_msg_distbuilder=['DistBuilder for the output messages.'...
                'In general one can use any DistBuilder. However, it makes '...
                'sense to use the DistBuilder corresponding to the type '...
                'of the variable to send to. Set to [] to do so.'];

            kv.prior_var = ['prior variance. Will not be optimized. Require > 0.'];
            kv.init_noise_var = ['initial point for the noise variance. ', ...
                'One for each output. Require > 0.'];
            kv.init_outer_width2 = ['initial point for the outer width squared ' ...
                'of the Gaussian kernel on mean embeddings.'];
            kv.init_embed_width2s = ['initial vector of embedding width squared ', ...
                '(widths for the inner kernel). One for each of the total stacked ' ...
                'dimensions.'];

            kv.num_inner_primal_features=['number of inner features.'];
            kv.num_primal_features=['number of actual random features to use ' ...
                'after a candidate FeatureMap is chosen'];
            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.out_msg_distbuilder=[]; % override in constructor
            st.prior_var = 1;
            st.init_noise_var = 1;
            st.init_outer_width2 = 1;
            st.init_embed_width2s = 1;

            st.num_inner_primal_features = 300;
            st.num_primal_features = 500;
            Op=Options(st);
        end

        % learn a DistMapper given the training data in MsgBundle.
        function [gm, C ]=learnDistMapper(this, bundle)
            assert(isa(bundle, 'MsgBundle'), 'input to constructor not a MsgBundle' );
            assert(bundle.count()>0, 'empty training set')
            oldRng = rng();
            rng(this.opt('seed'));

            outDa=bundle.getOutBundle();
            assert(isa(outDa, 'DistArray'));
            dout=outDa.get(1);
            assert(isa(dout, 'Distribution'));

            if ~this.hasKey('out_msg_distbuilder') || isempty(this.options.opt('out_msg_distbuilder'))
                % This ensures that the out_msg_distbuilder is of the same type 
                % as the output bundle in msgBundle. It can be different in general.
                this.options.opt('out_msg_distbuilder', dout.getDistBuilder());
            end
            % cell array of DistArray's
            inputDistArrays=bundle.getInputBundles();
            tensorIn=TensorInstances(inputDistArrays);

            op=this.options.toStruct();
            out_msg_distbuilder=op.out_msg_distbuilder;
            % learn operator
            outDa=bundle.getOutBundle();
            outStat=out_msg_distbuilder.getStat(outDa);

            % treat each output as a separate problem. p outputs.
            prior_var = this.opt('prior_var');
            init_noise_var = this.opt('init_noise_var');
            init_outer_width2 = this.opt('init_outer_width2');
            init_embed_width2s = this.opt('init_embed_width2s');
            num_inner_primal_features = this.opt('num_inner_primal_features');
            num_primal_features = this.opt('num_primal_features');

            p = size(outStat, 1);
            instancesMappers = cell(1, p);
            C = cell(1, p);
            dims=cellfun(@(da)unique([ da.d ]), tensorIn.instancesCell);

            % turn off some warning 
            warning_id = 'MATLAB:nearlySingularMatrix';
            warning('off', warning_id);
            for i=1:p
                outi = outStat(i, :);
                display(sprintf('Learning InstancesMapper for output %d', i));
                [noise_var, outer_width2, embed_width2s, loglik, exitflag, opinfo] = ...
                    RFGJointKGGGrad2MLLearner.learn_gp_marginal_likelihood(tensorIn, ...
                    outi, prior_var, init_noise_var, init_outer_width2, init_embed_width2s);
                % construct a FeatureMap from the parameters
                embed_width2s_cell = MatUtils.partitionArray(embed_width2s, dims);
                fm = RFGJointKGG(embed_width2s_cell, outer_width2, num_inner_primal_features, ...
                    num_primal_features);
                Op = BayesLinRegFM(fm, tensorIn, outi, noise_var );
                assert(isa(Op, 'UAwareInstancesMapper'));
                instancesMappers{i} = Op;

                cs = struct();
                cs.loglik = loglik;
                cs.exitflag = exitflag;
                cs.opinfo = opinfo;
                C{i} = cs;
            end
            im =  UAwareStackInsMapper(instancesMappers{:});
            gm = UAwareGenericMapper(im, out_msg_distbuilder, bundle.numInVars());

            warning('on', warning_id);
            rng(oldRng);
        end

        function s=shortSummary(this)
            s=mfilename;
        end
    end % end methods

    methods(Static)

        function KInvM = KInvTimesMat(R, M)
            % Compute K^-1*M where K = R'R (Cholesky factorization).
            % R is assumed to be upper triangular.
            % Quadratic time, not cubic.
            KInvM = linsolve(R, linsolve(R', M, struct('LT', true)), ...
            struct('UT', true));
        end

        function [noise_var, outer_width2, embed_width2s, loglik, exitflag, opinfo] = ...
            learn_gp_marginal_likelihood(tensorIn, Y, prior_var, ...
                init_noise_var, init_outer_width2, init_embed_width2s)

            jointDa = KGGaussianJoint.toJointDistArray(tensorIn.instancesCell);
            init_point = [init_noise_var, init_outer_width2, init_embed_width2s(:)']';
            objFun = @(log_opt_var) RFGJointKGGGrad2MLLearner.gp_loglik_wrapper(...
                log_opt_var, jointDa, Y, prior_var);

            op = struct();
            op.Diagnostics = 'on';
            op.DiffMaxChange = 1e3;
            op.Display = 'iter-detailed';
            %op.FinDiffType = 'central';
            op.FinDiffType = 'forward';
            op.FunValCheck = 'on';
            op.MaxIter = 100;
            op.TolFun = 1e-3;
            op.TolX = 1e-4;

            [x, loglik, exitflag, opinfo] = fminunc(objFun, log(init_point), op);
            % unpack
            noise_var = exp(x(1));
            outer_width2 = exp(x(2));
            embed_width2s = exp(x(3:end));

        end

        function [loglik ] = gp_loglik_obj(jointDa, Y, prior_var, noise_var, ...
                outer_width2, embed_width2s)
            % jointDa = joint DistArray formed by converting all incoming messages 
            % into one joint Gaussian.
            % based on GP for machine learning book, section 5.4.1
            assert(prior_var > 0);
            assert(noise_var > 0);

            % cap to avoid numerical instability
            noise_var = max(1e-9, noise_var);
            outer_width2 = max(1e-6, outer_width2);
            embed_width2s = max(1e-6, embed_width2s);

            assert(isnumeric(Y));
            Y = Y(:)';
            n = length(Y);
            assert(n == length(jointDa), 'Input-output lengths do not match.');

            % objective = log likelihood
            ker = KGGaussian(embed_width2s, outer_width2);

            Kappa = ker.eval(jointDa, jointDa);
            K = prior_var*Kappa + noise_var*eye(n);
            [R, p] = chol(K);
            if p > 1
                %error('Kernel matrix is not positive definite');
            end

            % alpha = K^-1 y'
            alpha = RFGJointKGGGrad2MLLearner.KInvTimesMat(R, Y');
            logDetK = 2*sum(log(diag(R)));
            loglik = -0.5*Y*alpha -0.5*logDetK - 0.5*n*log(2*pi);
            %% TODO: Can be improved without inv ?
            %KInv = RFGJointKGGGrad2MLLearner.KInvTimesMat(R, eye(n));

            %%-- gradient --
            %% noise_var gradient 
            %dK_noise_var = prior_var*Kappa + eye(n);
            %grad_noise_var = RFGJointKGGGrad2MLLearner.grad_dK(dK_noise_var, alpha, KInv);

            %% outer_width2 gradient 
            %D2  = -2*outer_width2*log(Kappa);
            %dK_outer_width2 = 0.5*Kappa.*D2/outer_width2^2;
            %grad_outer_width2 = RFGJointKGGGrad2MLLearner.grad_dK(dK_outer_width2, alpha, KInv);

        end

        function grad = grad_dK(dK_var, alpha, KInv)
            grad = 0.5*alpha'*dK_var*alpha - 0.5*KInv(:)'*dK_var(:);
        end

        function [f ] = gp_loglik_wrapper(log_opt_var, jointDa, Y, prior_var)
            % opt_var = (noise_var, outer_width2, embed_width2s)
            % To make the problem unconstrained, we optimize the log of the widths.
            %
            noise_var = exp(log_opt_var(1));
            outer_width2 = exp(log_opt_var(2));
            embed_width2s = exp(log_opt_var(3:end));
            [f ] = RFGJointKGGGrad2MLLearner.gp_loglik_obj(jointDa, Y, prior_var, noise_var, ...
                outer_width2, embed_width2s);
            f = -f;
            %grad = -grad;
        end

    end % end static methods

    
end

