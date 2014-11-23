classdef MPMapperLearner < DistMapperLearner
    %MPMAPPERLEARNER DistMapperLearner using on MatchingPursuit
    %   .

    properties(SetAccess=protected)
        % an instance of Options
        options;
    end

    methods 
        function this=MPMapperLearner()
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

            kv.separate_outputs = ['Treat each output as an independent problem.', ...
                'No parameter sharing between outputs. This increases the number of parameters.'];
            kv.mp_options = ['MatchingPursuit options struct.']
            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.out_msg_distbuilder=[]; 
            % options used in cond_ho_finiteout
            %
            st.separate_outputs = true;
            st.mp_options = [];

            Op=Options(st);
        end

        function [gm, Ress]=learnDistMapper(this, bundle )
            assert(isa(bundle, 'MsgBundle'), 'input not a MsgBundle' );
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
            % learn a DistMapper given the training data in MsgBundle.
            op=this.options.toStruct();

            % cell array of DistArray's
            inputDistArrays=bundle.getInputBundles();
            tensorIn=TensorInstances(inputDistArrays);
            out_msg_distbuilder=op.out_msg_distbuilder;
            % learn operator
            outDa=bundle.getOutBundle();
            outStat=out_msg_distbuilder.getStat(outDa);

            if this.opt('separate_outputs')
                % treat each output as a separate problem.
                p = size(outStat, 1);
                instancesMappers = cell(1, p);
                Ress = cell(1, p);
                % This can be parallelized
                for i=1:p
                    outi = outStat(i, :);
                    % MatchingPursuit implements InstancesMapper
                    mp = MatchingPursuit(tensorIn, outi);
                    mp_options = this.opt('mp_options');
                    mp.addOptions(mp_options);
                    display(sprintf('starting matching pursuit for input %d', i));
                    Res = mp.matchingPursuit();
                    Ress{i} = Res;

                    finalMp = mp.finalize();
                    instancesMappers{i} = finalMp;
                end
                im = StackInstancesMapper(instancesMappers{:});
                gm=GenericMapper(im, out_msg_distbuilder, bundle.numInVars());

            else
                mp = MatchingPursuit(tensorIn, outStat);
                mp_options = this.opt('mp_options');
                mp.addOptions(mp_options);
                Res = mp.matchingPursuit();
                Ress = Res;
                finalMp = mp.finalize();
                gm=GenericMapper(finalMp, out_msg_distbuilder, bundle.numInVars());
            end

        end

        function s=shortSummary(this)
            s=sprintf('%s', mfilename );
        end

    end % end methods

end

