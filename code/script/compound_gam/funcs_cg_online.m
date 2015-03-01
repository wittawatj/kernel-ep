function [ g ] = funcs_cg_online( )
%FUNCS_CG_ONLINE Functions to process the results of online learning on the compoung 
%gamma problem.

    g = struct();
    g.getFileStructs = @getFileStructs;
    g.plotInferenceResults = @plotInferenceResults;

end

function plotPosteriorShapeRateScatter(st)
    % plot agreement of posteriors between KJIT+Infer.NET and Infer.NET in a 
    % scatter plot.

    cutoff = min(1000, length(st.inShape));
    % only plot the correlation of the trials that the operator does not consult.
    Ikeep = ~st.consultOracle ;
    kepShape = st.postShapes(Ikeep);
    kepRate = st.postRates(Ikeep);
    dnetShape = st.oraPostShapes(Ikeep);
    dnetRate = st.oraPostRates(Ikeep);

    kepShape = kepShape(1:cutoff);
    kepRate = kepRate(1:cutoff);
    dnetShape = dnetShape(1:cutoff);
    dnetRate = dnetRate(1:cutoff);

    shapeCorr = corr2(log(kepShape), log(dnetShape));
    rateCorr = corr2(log(kepRate), log(dnetRate));

    figure
    hold on 
    %plot(dnetShape, dnetRate, '+k', 'LineWidth', 2);
    %plot(kepShape, kepRate, '*r', 'LineWidth', 2);
    plot(log(dnetShape), log(dnetRate), '+k', 'LineWidth', 2);
    plot(log(kepShape), log(kepRate), '*r', 'LineWidth', 1);
    set(gca, 'FontSize', 16);
    xlabel('Log shape parameter');
    ylabel('Log rate parameter')
    title(sprintf('Shape correlation: %.6g. Rate correlation: %.6g.',...
        shapeCorr, rateCorr));
    legend('Infer.NET', 'Infer.NET + KJIT');
    grid on
    hold off
end

function [klDnetKep, klKepDnet] = getPosteriorKLs(st, cutoff)
    trials = length(st.inShape);
    cutoff = min(cutoff, trials);
    klDnetKep = zeros(1, cutoff);
    klKepDnet = zeros(1, cutoff);

    for i=1:cutoff
        kepPost = DistGamma(st.postShapes(i), st.postRates(i));
        dnetPost = DistGamma(st.oraPostShapes(i), st.oraPostRates(i));

        klKepDnet(i) = kepPost.klDivergence(dnetPost);
        klDnetKep(i) = dnetPost.klDivergence(kepPost);
    end
end

function plotPosteriorKL(st)
    % plot agreement of posteriors between KJIT+Infer.NET and Infer.NET.

    trials = length(st.inShape);
    cutoff = min(400, trials);
    [klDnetKep, klKepDnet] = getPosteriorKLs(st, cutoff);

    timeSub = 1:cutoff;
    figure
    hold on 
    plot(timeSub, log(klDnetKep), '-b', 'LineWidth', 2);
    % If KL=0, log(KL)=-inf.
    I0 = klDnetKep==0;
    plot(find(I0), -20*ones(1, sum(I0)), 'r*');
    %plot(timeSub, log(klKepDnet), '-k', 'LineWidth', 2);
    set(gca, 'FontSize', 16);
    xlabel('Problems seen');
    ylabel('Log KL divergence')
    title('KL divergence of the posteriors')
    %legend('KL(Infer.NET || Infer.NET + KJIT)', 'KL(Infer.NET + KJIT || Infer.NET)');
    legend('KL(Infer.NET || Infer.NET + KJIT)', 'KL = 0');
    grid on
    hold off


end


function [st]= getFileStructs()
    % Load all variables into one struct.
    
    % size of observed incoming messages over which an initial batch learning is 
    % triggered.
    seed_to = 5000;
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
    timeSub = 1:min(1500, seed_to);
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
    window = 40;
    b = ones(1, window)/window;
    unSub = 1:min(1500, seed_to);
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

    %plotPosteriorKL(st);

    plotPosteriorShapeRateScatter(st);
end
