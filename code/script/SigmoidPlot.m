classdef SigmoidPlot  < handle
    %SigmoidPlot Investigate how much changing parameters of incoming messages
    %affects the outgoing messages in sigmoid factor problem.
    %   .

    properties 

    end
    methods(Static)
        function main()
            rng(10);

            % x (Gaussian), z (Beta)
            % messages from x 
            % messages from z 
            %z = DistBeta(3, 2);

            plotGroundTruth(mmx, vvx, z);
            %plotOperatorOuts(mmx, vvx, z);
        end

        function [mmx, vvx, z] = genData()
            z = DistBeta(2, 3);
            n =  40;
            mx = linspace(-16, 16, n);
            vx = linspace(0.5, 100, n);
            %vx = linspace(100, 300, n);

            mmx = repmat(mx, [length(vx), 1]);
            vvx = repmat(vx', [1, length(mmx)]);
        end


        function loadPlotLearnedFunc()
            fname = 'mp_distmapper_sigmoid_bw_proposal_10000_ntr8000.mat';
            %fname = 'mp_distmapper_sigmoid_bw_fixbeta_10000_ntr8000.mat';
            fpath = Expr.scriptSavedFile(fname);
            loaded = load(fpath, 'dm', 'trBundle', 'out_msg_distbuilder');

            trBundle = loaded.trBundle;
            dm = loaded.dm;
            out_msg_distbuilder = loaded.out_msg_distbuilder;

            Xtr = trBundle.getInputTensorInstances();
            Ytr = out_msg_distbuilder.getStat(trBundle.getOutBundle());

            true1 = Ytr(1, :);
            subI = randperm(length(true1), min(length(true1), 2000));
            outDa = dm.mapMsgBundle(trBundle);
            stat = out_msg_distbuilder.getStat(outDa);
            out1 = stat(1, :);

            true1 = true1(subI);
            out1 = out1(subI);

            % sort by true outputs.
            [strue, I] = sort(true1);
            sout1 = out1(I);
            n = length(I);

            figure
            hold on;
            set(gca, 'fontsize', 20);
            plot(1:n, strue, 'ro-', 'LineWidth', 2);
            plot(1:n, sout1,'bx-', 'LineWidth', 1 );
            legend('true output', 'MP learned function');
            hold off;
        end

        function loadPlotOperatorOuts()
            %fname = 'kernel_param_cmaes_RFGJointEProdLearner_sigmoid_bw_proposal_2000_2000.mat';
            %fname = 'distprop_learn_DistPropLearner_sigmoid_bw_proposal_2000_800.mat';
            %fname = 'ichol_learn_cmaes_ICholMapperLearner_sigmoid_bw_proposal_2000_800.mat';
            %fname = 'ichol_learn_laplace_ICholMapperLearner_sigmoid_bw_proposal_5000_5000.mat';
            %fname ='ichol_learn_laplace_ICholMapperLearner_sigmoid_bw_proposal_50000_50000.mat';
            %fname = 'ichol_learn_laplace_ICholMapperLearner_sigmoid_bw_proposal_10000_10000.mat';
            %fname = 'ichol_learn_cmaes_ICholMapperLearner_sigmoid_bw_proposal_25000_20000.mat';
            %fname ='ichol_learn_ICholMapperLearner_sigmoid_bw_proposal_10000_10000.mat';
            %fname ='ichol_learn_laplace_ICholMapperLearner_sigmoid_bw_proposal_5000_5000.mat';
            %fname ='ichol_learn_laplace_ICholMapperLearner_sigmoid_bw_proposal_10000_10000.mat';
            fname ='ichol_learn_laplace_ICholMapperLearner_sigmoid_bw_fixbeta_10000_10000.mat';
            sfile = Expr.scriptSavedFile(fname);
            loaded = load(sfile);
            s = loaded.s;
            dm = s.dist_mapper;
            SigmoidPlot.plotOperatorOuts(dm, fname);
        end

        function plotOperatorOuts(dm, dmIdentifier)
            if nargin < 2
                %dmIdentifier = dm.shortSummary();
                dmIdentifier = '';
            end
            [mmx, vvx, z] = SigmoidPlot.genData();
            inx = DistNormal(mmx(:)', vvx(:)');
            inxDa = DistArray(inx);
            n = length(inx);
            inz = DistBeta(repmat(z.alpha, 1, n), repmat(z.beta, 1, n));
            inzDa = DistArray(inz);
            douts = dm.mapDistArrays(inzDa, inxDa);

            tox_mean = reshape([douts.mean], size(mmx));
            tox_var = reshape([douts.variance], size(vvx));

            SigmoidPlot.plotMeanOutputs(mmx, vvx, tox_mean, z);
            superTitle=sprintf('%s. %s. Report outgoing mean to x. m_z = Beta(%.2g, %.2g).', dm.shortSummary(), dmIdentifier, z.alpha, z.beta );
            annotation('textbox', [0 0.9 1 0.1], ...
                'String', superTitle, ...
                'EdgeColor', 'none', ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 14, 'interpreter', 'none')
            SigmoidPlot.plotVarianceOutputs(mmx, vvx, tox_var, z);
            superTitle=sprintf('%s. %s. Report outgoing log variance to x. m_z = Beta(%.2g, %.2g).', dm.shortSummary(), dmIdentifier, z.alpha, z.beta );
            annotation('textbox', [0 0.9 1 0.1], ...
                'String', superTitle, ...
                'EdgeColor', 'none', ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 14, 'interpreter', 'none')
            %title(sprintf('Operator from \\text{%s}. Report outgoing mean to x. m_z = Beta(%.2g, %.2g).', fname, z.alpha, z.beta));
            %plotVarianceOutputs(mmx, vvx, tox_var, z);
            %title(sprintf('Operator from \\text{%s}. Report outgoing log variance to x. m_z = Beta(%.2g, %.2g).',fname, z.alpha, z.beta));
            %plotMeanOutputs(mmx, vvx, toz_mean, z);
            %title(sprintf('Operator from %s. Report outgoing mean to z. m_z = Beta(%.2g, %.2g).', fname, z.alpha, z.beta));
            %plotVarianceOutputs(mmx, vvx, toz_var, z);
            %title(sprintf('Operator from %s. Report outgoing log variance to z. m_z = Beta(%.2g, %.2g).', fname, z.alpha, z.beta));
        end

        function plotGroundTruth(mmx, vvx, z)

            %XX = DistNormal(mmx(:), vvx(:));
            %X = DistNormal(mx, vs);
            %z = DistBeta(5, 10);
            proposal = DistNormal(0, 15);
            fac = @(x)(1./(1+exp(-x)));
            N = length(mmx(:));
            % number of importance weights to draw
            K = 7e3;
            unif_f = -17;
            unif_t = 17;
            unif_length = unif_t - unif_f;

            toz_mean = zeros(size(mmx));
            toz_var = zeros(size(vvx));
            tox_builder = DistNormalBuilder();
            toz_builder = DistBetaBuilder();
            reports = 1:floor(N/10):N;
            xp = linspace(unif_f, unif_t, K);
            for i=1:N
                x = DistNormal(mmx(i), vvx(i));
                %xp = proposal.sampling0(K);
                zp = fac(xp);
                %W = x.density(xp).*z.density(zp)./proposal.density(xp);
                % uniform proposal. Don't need to divide.
                W = x.density(xp).*z.density(zp);
                WN = W/sum(W);

                xout = tox_builder.fromSamples(xp, WN);
                zout = toz_builder.fromSamples(zp, WN);

                tox_mean(i) = xout.mean;
                tox_var(i) = xout.variance;
                toz_mean(i) = zout.mean;
                toz_var(i) = zout.variance;
                if ismember(i, reports)
                    display(sprintf('%.2g %% completed', 100*i/N));
                end

            end

            tox_mean = reshape(tox_mean, size(mmx));
            tox_var = reshape(tox_var, size(vvx));
            toz_mean = reshape(toz_mean, size(mmx));
            toz_var = reshape(toz_var, size(vvx));

            SigmoidPlot.plotMeanOutputs(mmx, vvx, tox_mean, z);
            title(sprintf('Report outgoing mean to x. Fix incoming m_z = Beta(%.2g, %.2g).', z.alpha, z.beta));
            SigmoidPlot.plotVarianceOutputs(mmx, vvx, tox_var, z);
            title(sprintf('Report outgoing log variance to x. Fix incoming m_z = Beta(%.2g, %.2g).', z.alpha, z.beta));
            SigmoidPlot.plotMeanOutputs(mmx, vvx, toz_mean, z);
            title(sprintf('Report outgoing mean to z. Fix incoming m_z = Beta(%.2g, %.2g).', z.alpha, z.beta));
            SigmoidPlot.plotVarianceOutputs(mmx, vvx, toz_var, z);
            title(sprintf('Report outgoing log variance to z. Fix incoming m_z = Beta(%.2g, %.2g).', z.alpha, z.beta));
        end

        function plotMeanOutputs(mmx, vvx, to_mean, z)
            figure
            hold on
            C=contourf(mmx, log(vvx), to_mean, 20);
            %C=contourf(mmx./(vvx), -0.5./(vvx), to_mean, 20);
            clabel(C);
            colorbar
            set(gca, 'FontSize', 20);
            %xlabel('Gaussian input: m/v');
            %ylabel('Gaussian input -1/2v');
            xlabel('Gaussian input mean');
            ylabel('Gaussian input log variance'); 
            grid on
            hold off 
        end

        function plotVarianceOutputs(mmx, vvx, to_var, z)
            figure
            hold on
            C=contourf(mmx, log(vvx), log(to_var), 20);
            %C=contourf(mmx./(vvx), -0.5./(vvx), log(to_var), 10);
            clabel(C);
            colorbar
            set(gca, 'FontSize', 20);
            %xlabel('Gaussian input: m/v');
            %ylabel('Gaussian input -1/2v');
            xlabel('Gaussian input mean');
            ylabel('Gaussian input log variance');
            grid on
            hold off 
        end





    end

end

