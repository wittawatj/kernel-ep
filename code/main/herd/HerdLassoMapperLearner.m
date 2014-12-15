classdef HerdLassoMapperLearner < DistMapperLearner
    %HERDLASSOMAPPERLEARNER Learner a DistMapperLearner by herding weights with Lasso.
    %   - L1-penalized least squares for weights learning
    %
    properties(SetAccess=protected)
        % an instance of Options
        options;
    end

    methods
        function this=HerdLassoMapperLearner()
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
                'of the variable to send to. '];
            kv.cond_factor = ['conditional factor to consider. Mandatory.']
            kv.target_index = ['Index of the variable to be treated as the message', ... 
                ' sending taget.'];
            kv.max_locations = ['Maximum number of conditioning locations to ', ...
                'consider. = Max number of non-zero coefficients in Lasso. '];
            kv.num_lambda = ['Number of lambda (regularization parameter in ', ...
                'Lasso) to consider'];
            kv.lambdas = ['Lambda parameters to try']
            kv.cv_fold = ['#folds in cross validation'];
            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.out_msg_distbuilder=[]; 
            st.cond_factor = [];
            st.target_index = -1;
            st.max_locations = 5e3;
            st.num_lambda = 20;
            st.cv_fold = 3;
            st.lambdas = [];
            
            Op=Options(st);
        end

        function [gm, R]=learnDistMapper(this, bundle )
            assert(isa(bundle, 'MsgBundle'), 'input not a MsgBundle' );
            assert(bundle.count()>0, 'empty training set')

            outDa=bundle.getOutBundle();
            assert(isa(outDa, 'DistArray'));
            dout=outDa.get(1);
            assert(isa(dout, 'Distribution'));

            if this.isNoKeyOrEmpty('out_msg_distbuilder')
                % This ensures that the out_msg_distbuilder is of the same type 
                % as the output bundle in msgBundle. It can be different in general.
                this.options.opt('out_msg_distbuilder', dout.getDistBuilder());
            end

            n=length(bundle);
            % learn a DistMapper given the training data in MsgBundle.
            op=this.options.toStruct();
            [R, op] = herd_weights_l1(bundle, op);
            locSuffCell = R.locationSuffCell;
            locSuff = vertcat(locSuffCell{:});
            cond_points = R.cond_points;
            outLoc = R.outLocation;
            Lasso = R.Lasso;
            out_msg_distbuilder = this.opt('out_msg_distbuilder');

            instancesMappers = cell(1, length(Lasso));
            for j=1:length(Lasso)
                lambda_i = Lasso(j).FitInfo.IndexMinMSE;
                B = Lasso(j).B;
                F = Lasso(j).FitInfo;
                intercept = F.Intercept(lambda_i);
                weights = B(:, lambda_i)';

                instancesMappers{j} = HerdInstancesMapper(weights, intercept, ...
                    locSuff(j, :),  outLoc, cond_points);
            end

            im = StackInstancesMapper(instancesMappers{:});
            gm=GenericMapper(im, out_msg_distbuilder, bundle.numInVars());
        end

        function s=shortSummary(this)
            s=sprintf('%s', mfilename);
        end


    end

    methods(Static)

    end
end

