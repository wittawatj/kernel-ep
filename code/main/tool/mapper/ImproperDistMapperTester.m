classdef ImproperDistMapperTester < DistMapperTester
    %IMPROPERDISTMAPPERTESTER Plot mean/variance of improper messages and compare to the 
    %ground truth
    %    * This DistMapperTester is intended for checking improper output messages
    %    e.g., negative variance.
    %    * Only for 1d messages. Assume the output statistics are first 2 moments 
    %    in 1d.
    %
    
    properties(SetAccess=protected)
        % Instance of Options
        options;

        % DistMapperTester to test. From Distribution interface.
        distMapper;
    end
    
    methods
        function this=ImproperDistMapperTester(distMapper)
            this.distMapper=distMapper;
            this.options=this.getDefaultOptions();
        end

        function od=getOptionsDescription(this)
            % Return an instance of OptionsDescription describing possible options.
            % key-value pairs of open-description
            kv=struct();
            %kv.moment_distbuilder=['DistBuilder used to retrieve set of moments '...
                %'from the output messages. Set to [] to use the same type as the '...
                %'output messages. '];
            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            %st.moment_distbuilder=[];

            Op=Options(st);
        end

        function [impOut]=testDistMapper(this, testBundle)
            assert(isa(testBundle, 'MsgBundle'));
            outDa=this.distMapper.mapMsgBundle(testBundle);
            assert(isa(outDa, 'DistArray'));
            trueOutDa=testBundle.getOutBundle();
            assert(isa(trueOutDa, 'DistArray'));

            impOut=this.compareOutputs(trueOutDa, outDa);

        end

        function impOut=compareOutputs(this, trueOutDa, outDa)
            nte=outDa.count();
            % indices of improper output messages
            impInd=arrayfun(@(d)~d.isProper(), outDa.distArray);
            impOut=outDa.distArray(impInd);
            if isempty(impOut)
                display('No improper output messages.');
                return;
            end
            assert(isa(impOut, 'Distribution'));
            impn=length(impOut);

            % plot output means vs. true means
            figure
            outMeans=[impOut.mean];
            trueMeans=[trueOutDa.instances(impInd).mean];
            hold on
            set(gca, 'fontsize', 20);
            plot(outMeans, trueMeans, 'or', 'LineWidth', 1);
            legend('Mean')
            xlabel('Output means');
            ylabel('True means');
            title(sprintf('%d/%d improper output msgs (%s)', impn, nte, ...
                class(outDa.distArray(1) )));
            grid on
            %axis square
            hold off

            % plot output variances vs. true variances 
            figure
            outVars=[impOut.variance];
            trueVars=[trueOutDa.instances(impInd).variance];
            hold on
            set(gca, 'fontsize', 20);
            plot(outVars, trueVars, 'ob', 'LineWidth', 1);
            legend('Variance');
            xlabel('Output variances');
            ylabel('True variances');
            title(sprintf('%d/%d improper output msgs (%s)', impn, nte, ...
                class(impOut(1))));
            grid on
            %axis square
            hold off
        end

        % summary in string of this DistMapperTester
        function s=shortSummary(this)
            s=mfilename;
        end
    end
    
end

