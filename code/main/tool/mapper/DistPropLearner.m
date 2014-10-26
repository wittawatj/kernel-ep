classdef DistPropLearner < DistMapperLearner
    %DISTPROPLEARNER A DistMapperLearner using DistPropMap
    %   .
    
    properties(SetAccess=protected)
        % an instance of Options
        options;
    end
    
    methods
        function this=DistPropLearner()
            this.options = this.getDefaultOptions();
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

            kv.reglist=['list of regularization parameter candidates for ridge '...
                'regression.'];
            kv.use_multicore=['If true, use multicore package.'];

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.out_msg_distbuilder=[]; % override in constructor
            st.reglist=[1e-4, 1e-2, 1];
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

            op=this.options.toStruct();
            % cell array of DistArray's
            inputDistArrays=bundle.getInputBundles();
            tensorIn=TensorInstances(inputDistArrays);

            out_msg_distbuilder=op.out_msg_distbuilder;
            % learn operator
            outDa=bundle.getOutBundle();
            outStat=out_msg_distbuilder.getStat(outDa);

            daCell = bundle.getInputBundles();
            inputVarTypes = DistPropLearner.toInputVarTypes(daCell);
            % The featuremap_candidates is simply to use cond_fm_finiteout.
            % There is no candidate for DistPropLearner any way.
            % We still need reglist though.
            op.featuremap_candidates = {DistPropMap(inputVarTypes)};

            [Op, C]=CondFMFiniteOut.learn_operator(tensorIn, outStat, op);
            assert(isa(Op, 'InstancesMapper'));
            gm=GenericMapper(Op, out_msg_distbuilder, bundle.numInVars());

        end

        function s=shortSummary(this)
            s=mfilename;
        end

    end

    methods(Static)
        function inputVarTypes=toInputVarTypes(daCell)
            f = @(da)(class(da(1)));
            inputVarTypes = cellfun(f, daCell, 'UniformOutput', false);

        end

    end
end

