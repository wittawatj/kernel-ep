classdef RFGJointKGGLearner < DistMapperLearner
    %RFGJOINTKGGLEARNER A DistMapperLearner for RFGJointKGG. 
    %   - Use BayesLinRegFM.
    %   - Always treat each output separately because Bayesian linear regression 
    %   supports only real outputs.
    %    .

    properties(SetAccess=protected)
        % an instance of Options
        options;
    end

    methods
        function this=RFGJointKGGLearner()
            this.options=this.getDefaultOptions();
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';



            %kv.candidate_inner_primal_features = ['number of outer random features during ', 
            %'an actual training (not parameter selection phase).']
            %kv.candidate_primal_features=['number of random features to use for '...
            %    'candidate selection. During cross-validation, the number of '...
            %    'features can be low to be efficient.'];

            kv.num_inner_primal_features=['number of inner features during actual training.'];
            kv.num_primal_features=['number of actual random features to use ' ...
                'after a candidate FeatureMap is chosen'];

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

            kv.featuremap_candidates=['A cell array of FeatureMap.'...
                'This is mandatory. '];

            kv.reglist=['list of regularization parameter candidates for ridge '...
                'regression.'];
            kv.use_multicore=['If true, use multicore package.'];

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            %st.candidate_primal_features=600;
            %st.candidate_inner_primal_features = 300;
            % 1e4 random features 
            st.num_primal_features=2000;
            st.num_inner_primal_features = 1000;
            st.out_msg_distbuilder=[]; % override in constructor
            st.featuremap_candidates=[];
            % options used in cond_fm_finiteout
            st.reglist=[1e-2, 1, 100];
            st.use_multicore=true;

            Op=Options(st);
        end

        % learn a DistMapper given the training data in MsgBundle.
        function [gm, C]=learnDistMapper(this, bundle)
            assert(isa(bundle, 'MsgBundle'), 'input to constructor not a MsgBundle' );
            assert(bundle.count()>0, 'empty training set')

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
            if this.isNoKeyOrEmpty('featuremap_candidates')
                error('options featuremap_candidates must be set.');
            end

            % treat each output as a separate problem. p outputs.
            p = size(outStat, 1);
            instancesMappers = cell(1, p);
            for i=1:p
                outi = outStat(i, :);
                display(sprintf('Learning InstancesMapper for output %d', i));
                [Op, C] = BayesLinRegFM.learn_operator(tensorIn, outi, op);
                assert(isa(Op, 'UAwareInstancesMapper'));
                instancesMappers{i} = Op;
            end
            im =  UAwareStackInsMapper(instancesMappers{:});
            gm = UAwareGenericMapper(im, out_msg_distbuilder, bundle.numInVars());

        end

        function s=shortSummary(this)
            s=mfilename;
        end


    end

    methods(Static)

    end

end

