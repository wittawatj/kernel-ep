classdef RFGMVMapperLearner < DistMapperLearner
    %RFGMVMAPPERLEARNER A DistMapperLearner which learns a DistMapper using 
    %RandFourierGaussMVMap.
    %   .
    
    properties(SetAccess=protected)
        % an instance of Options
        options;

        % training MsgBundle
        trainBundle;
    end
    
    methods
        function this=RFGMVMapperLearner(msgBundle)
            assert(isa(msgBundle, 'MsgBundle'), 'input to constructor not a MsgBundle' );
            assert(msgBundle.count()>0, 'empty training set')
            this.trainBundle=msgBundle;
            this.options=this.getDefaultOptions();

            outDa=msgBundle.getOutBundle();
            assert(isa(outDa, 'DistArray'));
            dout=outDa.get(1);
            assert(isa(dout, 'Distribution'));
            
            % This ensures that the out_msg_distbuilder is of the same type 
            % as the output bundle in msgBundle. It can be different in general.
            this.options.opt('out_msg_distbuilder', dout.getDistBuilder());
            
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';
            
            kv.med_subsamples= ['number of samples to be used for computing pairwise '...
                'median distance. Subsampling makes it more efficient as an ' ...
                'accurate median is not needed.'];
            
            % Numerical array of factors to be multiplied with the median
            % distance on means. Used to generate FeatureMap candidates.
            kv.mean_med_factors=['numerical array of scaling factors to be ' ...
                'multiplied with the median distance on means.'];
            
            % Used to generate FeatureMap candidates.
            kv.variance_med_factors = ['numerical array of scaling factors to be ' ...
                'multiplied with the median distance on variances.'];            

            % number of primal features to use for candidates. 
            % This number is not necessarily the same during the test time.
            % Typically during LOOCV, the number of features can be low to make
            % it fast to select a candidate.
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
                'If mean_med_factors and variance_med_factors '...
                'are set, and featuremap_candidates'...
                'is set to [], median distance heuristic will be used to automatically '...
                'generate a cell array of candidates.' ];
            
            kv.reglist=['list of regularization parameter candidates for conditional '... 
            'mean embedding operator'];
            kv.use_multicore=['If true, use multicore package.'];

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.med_subsamples=1500;
            st.mean_med_factors=[1/3, 1, 3];
            st.variance_med_factors=[1/3, 1, 3];
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
        function dm=learnDistMapper(this )
            op=this.options.toStruct();
            bundle=this.trainBundle;
            % cell array of DistArray's
            inputDistArrays=bundle.getInputBundles();
            tensorIn=TensorInstances(inputDistArrays);

            if ~(isfield(op, 'featuremap_candidates') ...
                    && ~isempty(op.featuremap_candidates))
                % featuremap_candidates not set or set to []
                % compute ones

                % FeatureMap candidates
                mean_med_factors=op.mean_med_factors;
                variance_med_factors=op.variance_med_factors;
                assert(all(mean_med_factors>0));
                assert(all(variance_med_factors>0));

                candidate_primal_features=op.candidate_primal_features;
                assert(candidate_primal_features>0);
                med_subsamples=op.med_subsamples;
                assert(med_subsamples>0);
                FMcell=RandFourierGaussMVMap.candidates(tensorIn, mean_med_factors, ...
                    variance_med_factors, candidate_primal_features, med_subsamples);
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
            dm=GenericMapper(Op, out_msg_distbuilder, bundle.numInVars());

        end

        function s=shortSummary(this)
            s=mfilename;
        end

        function s=saveobj(this)
            s.trainBundle=this.trainBundle;
            s.options=this.options;
        end


    end

    methods(Static)
        function obj=loadobj(s)
            obj=RFGMVMapperLearner(s.trainBundle);
            obj.options=s.options;
        end
    end

    
end

