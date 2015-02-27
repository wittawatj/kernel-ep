function [ g ] = funcs_cg_online( )
%FUNCS_CG_ONLINE Functions to process the results of online learning on the compoung 
%gamma problem.

    g = struct();
    g.getFileStructs = @getFileStructs;
    g.plotInferenceResults = @plotInferenceResults;

end


function plotPosteriorKL(kepCells, isCells)
    % plot agreement of posteriors between KJIT+Infer.NET and Infer.NET.

    kepPostMeans = [];
    kepPostCovs = [];
    isPostMeans = [];
    isPostCovs =[];

    klIsKep = zeros(1, length(kepCells));
    klDnetKep = zeros(1, length(kepCells));
    klDnetIs = zeros(1, length(kepCells));

    for i=1:length(kepCells)
        kep = kepCells{i};
        km = kep.postMeans{1};
        kc = kep.postCovs{1};
        kepPostMeans = [kepPostMeans, km];
        kepPostCovs = cat(3, kepPostCovs, kc );
        kepPost = DistNormal(km, kc);

        is = isCells{i};
        ism = is.postMeans{1};
        isc = is.postCovs{1};
        isPostMeans = [isPostMeans, ism];
        isPostCovs = cat(3, isPostCovs, isc);
        isPost = DistNormal(ism, isc);

        dm = kep.dnetPostMeans{1};
        dc = kep.dnetPostCovs{1};
        dnetPost = DistNormal(dm, dc);

        klIsKep(i) = isPost.klDivergence(kepPost);
        klDnetKep(i) = dnetPost.klDivergence(kepPost);
        klDnetIs(i) = dnetPost.klDivergence(isPost);

    end
    %kepPost = DistNormal(kepPostMeans, kepPostCovs);
    %isPost = DistNormal(isPostMeans, isPostCovs);

    seeds = 1:length(kepCells);
    figure
    hold on 
    plot(seeds, log(klIsKep), '-r', 'LineWidth', 2);
    plot(seeds, log(klDnetKep), '-b', 'LineWidth', 2);
    plot(seeds, log(klDnetIs), '-k', 'LineWidth', 2);
    set(gca, 'FontSize', 18);
    xlabel('Problems seen');
    ylabel('Log KL divergence')
    title('KL divergence of the posteriors')
    legend('KL(sampling, KJIT)', 'KL(Infer.NET, KJIT)', 'KL(Infer.NET, sampling)');
    grid on
    hold off


end

function [st]= getFileStructs()
    % Load all variables into one struct.
    
    % size of observed incoming messages over which an initial batch learning is 
    % triggered.
    seed_to = 1000;
    initBatchTriggerSize = 20;
    epIter = 10;

    scriptFol = Global.getScriptFolder();
    fullFileFunc = @(fn)fullfile(scriptFol, 'logistic_msg', 'infer.net_saved', fn);
    fname = sprintf('kjit_cg_iter%d_bt%d_st%d.mat', epIter, initBatchTriggerSize, seed_to);
    fpath = fullFileFunc(fname);
    display(sprintf('Loading %s', fpath));
    st = load(fpath);
    % Struct example: 
    % st = 
    %      inShape: [300x1 double]
    %       inRate: [300x1 double]
    %     outShape: [300x1 double]
    %      outRate: [300x1 double]
    %consultOracle: [300x1 double]
    %  uncertainty: [2x300 double]
    %  oraOutShape: [300x1 double]
    %   oraOutRate: [300x1 double]
    %   inferTimes: [300x1 double]
    %oraInferTimes: [300x1 double]
    %   postShapes: [300x1 double]
    %    postRates: [300x1 double]
    %oraPostShapes: [300x1 double]
    % oraPostRates: [300x1 double]
    %           Ns: [300x1 double]
    %   trueRate2s: [300x1 double]
    %    truePrecs: [300x1 double]

end


function plotInferenceResults()
    st = getFileStructs();
    seed_to = length(st.inShape);

    % plot inference time.
    timeSub = 1:min(500, seed_to);
    figure
    hold on 
    plot(timeSub, log(st.oraInferTimes(timeSub)), '-k', 'LineWidth', 2);
    plot(timeSub, log(st.inferTimes(timeSub)), '-r', 'LineWidth', 2);
    set(gca, 'FontSize', 14);
    ylabel('Time in log(ms)')
    xlabel('Problems seen');
    title('Inference time')
    legend('Infer.NET', 'Sampling + KJIT');
    grid on
    hold off

    % uncertaintyOutX over each time point 
    figure
    hold on 
    % 1 for predicing the mean of X
    window = 20;
    b = ones(1, window)/window;
    unSub = 1:min(500, seed_to);
    % uncertainty for the first output which is log(shape)
    Un = max(st.uncertainty(1, :), st.uncertainty(2, :));
    unFil = filter(b, 1, Un);
    plot( Un(1, unSub), '-b', 'LineWidth', 1);
    plot( unFil(1, unSub), '-r', 'LineWidth', 2);
    % plot consultations
    consultTimesSub = find(st.consultOracle(unSub));
    plot( consultTimesSub, Un(1, consultTimesSub), '^k' );
    xlim([1, length(unSub)]);
    set(gca, 'FontSize', 10);
    ylabel('Log predictive variance');
    xlabel('Time for each input');
    title('Predictive variance of the incoming message at each time point')
    legend('Log predictive variance', sprintf('Moving average'), 'Consult oracle');
    grid on
    hold off

    %plotPosteriorKL(kepCells, isCells);

end
