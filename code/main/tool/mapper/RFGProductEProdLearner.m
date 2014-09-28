classdef RFGProductEProdLearner < DistMapperLearner
    %RFGPRODUCTEPRODLEARNER DistMapperLearner for RFGProductEProdMap

    properties(SetAccess=protected)
        % an instance of Options
        options;

    end

    methods
        function this=RFGProductEProdLearner()
            this.options=this.getDefaultOptions();

        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';

            kv.med_subsamples= ['number of samples to be used for computing pairwise '...
                'median distance. Subsampling makes it more efficient as an ' ...
                'accurate median is not needed.'];

            kv.med_factors=['numerical array of scaling factors to be ' ...
                'multiplied with the median distance heuristic. '...
                'Used to generate candidates'];

            kv.candidate_primal_features=['number of random features to use for '...
                'candidate selection. During cross-validation, the number of '...
                'features can be low to be efficient.'];

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
                'This is not expected to be set directly.'...
                'If med_factors '...
                'is set, and featuremap_candidates'...
                'is set to [], median distance heuristic will be used to automatically '...
                'generate a cell array of candidates.' ];

            kv.reglist=['list of regularization parameter candidates for ridge '...
                'regression.'];
            kv.use_multicore=['If true, use multicore package.'];

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.med_subsamples=1500;
            st.med_factors=[1/10, 1, 10];
            st.candidate_primal_features=2000;
            % 1e4 random features 
            st.num_primal_features=1e4;
            st.out_msg_distbuilder=[]; % override in constructor
            st.featuremap_candidates=[];
            % options used in cond_fm_finiteout
            st.reglist=[1e-2, 1, 100];
            st.use_multicore=true;

            Op=Options(st);
        end

        % learn a DistMapper given the training data in MsgBundle.
        function [gm, C]=learnDistMapper(this, bundle )
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

            op=this.options.toStruct();
            % cell array of DistArray's
            inputDistArrays=bundle.getInputBundles();
            tensorIn=TensorInstances(inputDistArrays);

            if ~(isfield(op, 'featuremap_candidates') ...
                    && ~isempty(op.featuremap_candidates))
                % featuremap_candidates not set or set to []
                % compute ones

                % FeatureMap candidates
                med_factors=op.med_factors;
                assert(all(med_factors>0));

                candidate_primal_features=op.candidate_primal_features;
                assert(candidate_primal_features>0);
                med_subsamples=op.med_subsamples;
                assert(med_subsamples>0);
                FMcell=RFGProductEProdMap.candidates(tensorIn, med_factors, ...
                    candidate_primal_features, med_subsamples);
                % set to options
                this.opt('featuremap_candidates', FMcell);
            end

            op=this.options.toStruct();
            out_msg_distbuilder=op.out_msg_distbuilder;
            % learn operator
            outDa=bundle.getOutBundle();
            outStat=out_msg_distbuilder.getStat(outDa);
            [Op, C]=CondFMFiniteOut.learn_operator(tensorIn, outStat, op);
            assert(isa(Op, 'InstancesMapper'));
            gm=GenericMapper(Op, out_msg_distbuilder, bundle.numInVars());

        end

        function s=shortSummary(this)
            s=mfilename;
        end


    end
    
end

