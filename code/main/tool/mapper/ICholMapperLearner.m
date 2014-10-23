classdef ICholMapperLearner < DistMapperLearner
    %ICHOLMAPPERLEARNER DistMapperLearner based on incomplete Cholesky factorization.
    %    * This DistMapperLearner works on any Kernel.
    %    * The data is of type TensorInstances 
    %    * Kernel is usually a KProduct
    %
    
    properties(SetAccess=protected)
        % an instance of Options
        options;
    end

    methods
        function this=ICholMapperLearner()
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
            % Default to the same type as Out array.
            kv.out_msg_distbuilder=['DistBuilder for the output messages.'...
                'In general one can use any DistBuilder. However, it makes '...
                'sense to use the DistBuilder corresponding to the type '...
                'of the variable to send to. Set to [] to do so.'];

            kv.kernel_candidates=['A cell array of Kernels to be selected '...
                'by repeated holdouts. Make sure that the Kernels are compatible '...
                'with the data Instances. This must not be empty.'];
            kv.num_ho=['number of repeated holdouts to perform. Train h times '...
                'on different randomly drawn h sets and test on a separete '...
                'nonoverlapping test set.'];

            kv.ho_train_size=['Training size (#samples) in the repeated holdouts'];
            kv.ho_test_size=['Test size (#samples) in the repeated holdouts'];
            kv.chol_tol=['Tolerance for incomplete Cholesky on kernel matrix.'];
            kv.chol_maxrank=['Maximum incomplete Cholesky rank. #rows of R in '...
                'K~R''R.'];
            kv.chol_maxrank_train = ['Maximum incomplete Cholesky rank for training.']
            kv.reglist=['list of regularization parameter candidates for ridge '...
                'regression.'];
            kv.use_multicore=['If true, use multicore package.'];
            kv.use_cmaes = ['True to use cma-es black-box optimization for parameter tuning'];
            kv.kernel_mode = ['Kernel mode. Only matter if use_cmaes = true. See cond_ho_finiteout_cmaes.']
            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.out_msg_distbuilder=[]; % override in constructor
            % options used in cond_ho_finiteout
            %
            % Must be manually set
            st.kernel_candidates=[]; 
            st.num_ho=5;
            st.chol_tol=1e-8;
            st.chol_maxrank=600;
            st.chol_maxrank_train=100;
            st.reglist=[1e-2, 1, 100];
            st.use_multicore=true;
            st.use_cmaes = false;

            Op=Options(st);
        end

        function [gm, C]=learnDistMapper(this, bundle )
            assert(isa(bundle, 'MsgBundle'), 'input to constructor not a MsgBundle' );
            assert(bundle.count()>0, 'empty training set')

            outDa=bundle.getOutBundle();
            assert(isa(outDa, 'DistArray'));
            dout=outDa.get(1);
            assert(isa(dout, 'Distribution'));

            if ~this.options.hasKey('out_msg_distbuilder') || isempty(this.options.opt('out_msg_distbuilder'))
                % This ensures that the out_msg_distbuilder is of the same type 
                % as the output bundle in msgBundle. It can be different in general.
                this.options.opt('out_msg_distbuilder', dout.getDistBuilder());
            end

            n=length(bundle);
            if ~this.options.hasKey('ho_train_size')
                st.ho_train_size=floor(0.7*n);
            end
            if ~this.options.hasKey('ho_test_size')
                st.ho_test_size=floor(0.3*n);
            end
            % learn a DistMapper given the training data in MsgBundle.
            op=this.options.toStruct();

            % cell array of DistArray's
            inputDistArrays=bundle.getInputBundles();
            tensorIn=TensorInstances(inputDistArrays);
            out_msg_distbuilder=op.out_msg_distbuilder;
            % learn operator
            outDa=bundle.getOutBundle();
            outStat=out_msg_distbuilder.getStat(outDa);
            op.ho_train_size=op.ho_train_size;
            op.ho_test_size=op.ho_test_size;
            assert(op.ho_train_size+op.ho_test_size<=n, '#Train and test samples exceed total samples');

            if this.opt('use_cmaes')
                %op.kernel_mode = this.opt('kernel_mode');
                [Op, C]=CondCholFiniteOut.learn_operator_cmaes(tensorIn, outStat, op);
            else
                if ~(isfield(op, 'kernel_candidates') ...
                        && ~isempty(op.kernel_candidates))
                    error('option kernel_candidates must be set.')
                end
                [Op, C]=CondCholFiniteOut.learn_operator(tensorIn, outStat, op);
            end
            assert(isa(Op, 'InstancesMapper'));
            gm=GenericMapper(Op, out_msg_distbuilder, bundle.numInVars());

        end

        function s=shortSummary(this)
            s=sprintf('%s(maxrank=%d)', mfilename, this.opt('chol_maxrank'));
        end


    end

    methods(Static)

    end
end

