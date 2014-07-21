classdef DivDistMapperTester < DistMapperTester
    %DIVDISTMAPPERTESTER A DistMapperTester to plot the histogram of divergence 
    %values to the groundtruth messages.
    %    Plot a histogram like in Figure 1 of 
    %    “Learning to Pass Expectation Propagation Messages.”
    %    * Work only for 1d Distribution because the second plot will show sample 
    %    plots of output distributions from different locations of the histogram.
    %

    properties(SetAccess=protected)
        % Instance of Options
        options;

        % DistMapperTester to test. From Distribution interface.
        distMapper;
    end

    methods
        
        function this=DivDistMapperTester(distMapper)
            this.distMapper=distMapper;
            this.options=this.getDefaultOptions();
        end

        function od=getOptionsDescription(this)
            % Return an instance of OptionsDescription describing possible options.
            % key-value pairs of open-description
            kv=struct();
            %kv.seed='random seed';
            kv.improper_dist_div=['divergence values to be used if '...
               'the output message is not isProper(). ']; 
            kv.div_function=['divergence function to use. Choices: {KL, Hellinger}'];
            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            % maximum penalty if the output message is impropper.
            st.improper_dist_div=1;

            st.div_function='KL';
            Op=Options(st);
        end

        function [Divs ]=getDivergence(this, outDa, trueOutDa)
            % outDa is the output DistArray from the DistMapper
            % trueOutDa is the groundtruth DistArray

            % n test
            nte = outDa.count();
            Divs = zeros(1, nte);
            % test the operator on the training set.
            for i=1:nte
                qout=outDa.get(i);
                assert(isa(qout, 'Distribution'));
                assert(isa(qout, 'HasHellingerDistance'), ...
                    'Output Distribution needs to implement HasHellingerDistance.');
                qtrue=trueOutDa.get(i);
                assert(isa(qtrue, 'Distribution'));
                assert(isa(qtrue, 'HasHellingerDistance'), ...
                    'Output Distribution needs to implement HasHellingerDistance.');

                if qout.isProper() && qtrue.isProper()
                    div=this.opt('div_function');
                    if strcmp(div, 'KL') 

                        hl = qtrue.klDivergence(qout);
                    elseif strcmp(div, 'Hellinger')
                        hl = qtrue.distHellinger(qout);
                    else 
                        error('Unknown value for div_function options');
                    end
                else
                    hl=nan();

                end
                Divs(i) = hl;
            end
        end

        function [Divs, outDa]=testDistMapper(this, testBundle)
            % plot histogram of divergence values.
            %
            assert(isa(testBundle, 'MsgBundle'));
            outDa=this.distMapper.mapMsgBundle(testBundle);
            assert(isa(outDa, 'DistArray'));
            trueOutDa=testBundle.getOutBundle();
            assert(isa(trueOutDa, 'DistArray'));
            Divs=this.getDivergence(outDa, trueOutDa);
            imCount=sum(isnan(Divs));
            %improperPenalty=this.opt('improper_dist_div');
            %Divs(isnan(Divs))=improperPenalty;

            figure
            hold on
            logDivs=log(Divs);
            keepI=isfinite(logDivs) & imag(logDivs)==0;
            logDivs=logDivs(keepI);

            hist(logDivs, 20);
            set(gca, 'fontsize', 20);
            xlabel(sprintf('Log divergence of %s', class(outDa.get(1)) ));
            ylabel('Frequency');
            title(sprintf('%d/%d improper messages. Mean: %.3f, SD: %.3f',...
                imCount, outDa.count(), mean(logDivs), std(logDivs)));
            grid on;
            hold off

            trueOutDa=trueOutDa.instances(keepI);
            outDa=outDa.instances(keepI);

            % sort logDivs. 
            [soLogDivs, I]=sort(logDivs);
            soTrueOutDa=trueOutDa.instances(I);
            soOutDa=outDa.instances(I);

            nte=soOutDa.count();
            % Take the best, 2 median, and worst
            % green=predicted, red=ground truth
            figure
            %title(sprintf('Output messages: %s', class(outDa.get(1))));
            % best
            subplot(2, 2, 1);
            this.plotOutputPairs(soOutDa.get(1), soTrueOutDa.get(1), ...
                sprintf('Best prediction. Div: %.3f', soLogDivs(1)));

            % median 1
            subplot(2, 2, 2);
            m1=floor(nte/2);
            this.plotOutputPairs(soOutDa.get(m1), soTrueOutDa.get(m1), ...
                sprintf('Median prediction 1. Div: %.3f', soLogDivs(m1)));

            % median 2
            subplot(2, 2, 3);
            m2=m1+1;
            this.plotOutputPairs(soOutDa.get(m2), soTrueOutDa.get(m2), ...
                sprintf('Median prediction 2. Div: %.3f', soLogDivs(m2)));

            % worst 
            subplot(2, 2, 4);
            this.plotOutputPairs(soOutDa.get(nte), soTrueOutDa.get(nte), ...
                sprintf('Worst prediction. Div: %.3f', soLogDivs(nte)));

        end

        function plotOutputPairs(this, predictDist, trueDist, titleText)
            % plot the two Distributions 
            % This function works only for 1d Distribution's
            assert(isa(predictDist, 'Distribution'))
            assert(isa(trueDist, 'Distribution'))
            % Distribution implies Density
            sd=max([predictDist.variance, trueDist.variance])^0.5;
            center=mean( [predictDist.mean, trueDist.mean] );

            xrange=linspace(center-3*sd, center+3*sd, 2e4);
            hold on 
            plot(xrange, predictDist.density(xrange), 'g-', 'LineWidth', 2);
            plot(xrange, trueDist.density(xrange), 'r-', 'LineWidth', 2);
            legend('predicted', 'ground truth');
            set(gca, 'fontsize', 18);
            title(titleText);
            grid on
            hold off
        end

        function [Divs, outDa]=plotMeanVarianceErrors(this, testBundle)

            assert(isa(testBundle, 'MsgBundle'));
            outDa=this.distMapper.mapMsgBundle(testBundle);
            assert(isa(outDa, 'DistArray'));
            trueOutDa=testBundle.getOutBundle();
            assert(isa(trueOutDa, 'DistArray'));
            Divs=this.getDivergence(outDa, trueOutDa);

            improperPenalty=this.opt('improper_dist_div');
            Divs(isnan(Divs))=improperPenalty;
            % plot to compare groundtruth and output messages
            % means
            TrMeans = [trueOutDa.mean];
            OpMeans = [outDa.mean];
            hmean=figure;
            hold on
            set(gca, 'fontsize', 20);
            stem(TrMeans, 'or');
            stem(OpMeans, 'ob');
            plot( abs(TrMeans-OpMeans), '-k', 'LineWidth', 2);
            xlabel('Message index');
            ylabel(sprintf('Mean of %s', class(outDa.get(1)) ));
            % title(sprintf('Training size: %d', n));
            legend('True means', 'Output means', 'abs. diff.');
            grid on
            hold off

            % variance
            TrVar = [trueOutDa.variance];
            OpVar = [outDa.variance];
            hvar= figure;
            hold on
            set(gca, 'fontsize', 20);
            stem(TrVar, 'or');
            stem(OpVar, 'ob');
            plot( abs(TrVar-OpVar), '-k', 'LineWidth', 2);
            xlabel('Message index');
            ylabel(sprintf('Variance of %s', class(outDa.get(1)) ));
            % title(sprintf('Training size: %d', n));
            legend('True variance', 'Output variance', 'abs. diff.');
            grid on
            hold off

            % plot divergences 
            hhell=figure;
            hold on
            stem(Divs);
            set(gca, 'fontsize', 20);
            title(sprintf('Divergence on %s', class(outDa.get(1))) );
            xlabel('Messsage index');
            ylabel('Divergence');
            ylim([0, 1]);
            grid on
            hold off

        end

        % summary in string of this DistMapperTester
        function s=shortSummary(this)
            s=mfilename;
        end
    end

    methods (Static)
    end

end

