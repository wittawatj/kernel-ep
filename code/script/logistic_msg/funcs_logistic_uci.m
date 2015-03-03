function [ g] = funcs_logistic_uci( )
%FUNCS_LOGISTIC_UCI Functions to process results of logistic regression on 
%UCI datasets.

    g = struct();

    g.getFileStructs = @getFileStructs;
    g.plotTemporalUncertainty = @plotTemporalUncertainty;
    g.plotInferenceResults = @plotInferenceResults;
    g.plotIncomingMessages = @plotIncomingMessages;
end

function xProblemInds= getProblemIndices(kjit)
    % an array of time points of the beginning of new problems
    %
    dataNames = fieldnames(kjit);
    prevProblemI = 1;
    xProblemInds = [];
    for i=1:length(dataNames)
        dn = dataNames{i};
        dataResult = kjit.(dn);
        xProblemInds(end+1) = prevProblemI;
        prevProblemI = prevProblemI + length(dataResult.consultOracleOutX);
    end
end

function uncertaintyOutX = getAllUncertaintyOutX(kjit)

    uncertaintyOutX = [];
    dataNames = fieldnames(kjit);
    for i=1:length(dataNames)
        dn = dataNames{i};
        dataResult = kjit.(dn);
        uncertaintyOutX = [uncertaintyOutX, dataResult.uncertaintyOutX];
    end
end

function consultationX = getAllConsultations(kjit)

    consultationX = [];
    dataNames = fieldnames(kjit);
    for i=1:length(dataNames)
        dn = dataNames{i};
        dataResult = kjit.(dn);
        consultationX = [consultationX, dataResult.consultOracleOutX(:)'];
    end
end

function plotPosteriorKL(kjit, is, dnet)
    % plot agreement of posteriors between sampling vs. Infer.NET.
    % KJIT vs sampling.

    [dataNames, dataLabels, testsetFNames] = getDataNames();

    klIsKep = zeros(1, length(dataNames));
    klDnetKep = zeros(1, length(dataNames));
    klDnetIs = zeros(1, length(dataNames));

    for i=1:length(dataNames)
        dn = dataNames{i};

        kjitMean = kjit.(dn).postMeans{1};
        kjitCov = kjit.(dn).postCovs{1};
        kjitPost =DistNormal(kjitMean, kjitCov);

        isMean = is.(dn).postMeans{1};
        isCov = is.(dn).postCovs{1};
        isPost = DistNormal(isMean, isCov);

        dnetMean = dnet.(dn).dnetPostMean;
        dnetCov = dnet.(dn).dnetPostCov;
        dnetPost = DistNormal(dnetMean, dnetCov);

        klIsKep(i) = isPost.klDivergence(kjitPost);
        klDnetKep(i) = dnetPost.klDivergence(kjitPost);
        klDnetIs(i) = dnetPost.klDivergence(isPost);

    end

    E = [klIsKep; klDnetKep; klDnetIs]';

    figure
    hold on 

    H = bar(log(E), 1, 'grouped');
    set(H(1), 'FaceColor', [0.8, 0, 0]);
    set(H(2), 'FaceColor', [0, 0, 0.8]);
    set(H(3), 'FaceColor', [0, 0, 0]);
    set(gca, 'FontSize', 12);
    legend('KL(sampling || KJIT)', 'KL(Infer.NET || KJIT)', 'KL(Infer.NET || sampling)');
    set(gca, 'XTick', 1:length(dataLabels));
    set(gca, 'XTickLabel', dataLabels);
    title('KL divergence of the posteriors')
    grid on
    hold off

end

function loss=classWeighted01Loss(Y, Yhat)
    % weight class 1 with (1- proportion of class 1) and vice versa
    assert(length(Y) == length(Yhat));
    prop0 = mean(Y==0);
    prop1 = mean(Y==1);
    I0 = Y==0;
    I1 = ~I0;
    n = length(Y);
    %z = 1.0/(1/prop0 + 1/prop1);
    %loss =  z* ( mean(Y(I0)~=Yhat(I0))./prop0 + mean(Y(I1)~=Yhat(I1))./prop1 );
    %loss =  ( sum(Y(I0)~=Yhat(I0))*0.5/prop0/n + sum(Y(I1)~=Yhat(I1))*0.5./prop1/n );
    loss = mean(Y~=Yhat);

end

function  plot01Loss(kjit, is, dnet)
    %. Plot 0-1 loss. Test on a separate test data.

    [dataNames, dataLabels, testsetFNames] = getDataNames();
    sigFunc = @(p)1./(1+ exp(-p));
    kjitLoss = zeros(1, length(dataNames));
    sLoss = zeros(1, length(dataNames));
    dnetLoss = zeros(1, length(dataNames));

    scriptFol = Global.getScriptFolder();
    fpathFunc = @(fn)fullfile(scriptFol, 'logistic_msg', ...
        'infer.net_saved', 'data', fn);
    for i=1:length(dataNames)
        dn = dataNames{i};
        % Expect dxn X and 1xn Y
        testDat = load(fpathFunc(testsetFNames{i}));
        kjitMean = kjit.(dn).postMeans{1};
        kjitCov = kjit.(dn).postCovs{1};
        Pkjit = sigFunc(kjitMean(:)'*testDat.X);
        %kjitLoss(i) = mean(round(Pkjit) ~= testDat.Y);
        kjitLoss(i) = classWeighted01Loss(round(Pkjit), testDat.Y);

        isMean = is.(dn).postMeans{1};
        isCov = is.(dn).postCovs{1};
        Pis = sigFunc(isMean(:)'*testDat.X);
        %isLoss(i) = mean(round(Pis) ~= testDat.Y);
        isLoss(i) = classWeighted01Loss(round(Pis), testDat.Y);

        dnetMean = dnet.(dn).dnetPostMean;
        dnetCov = dnet.(dn).dnetPostCov;
        Pdnet = sigFunc(dnetMean(:)'*testDat.X);
        %dnetLoss(i) = mean(round(Pdnet) ~= testDat.Y);
        dnetLoss(i) = classWeighted01Loss(round(Pdnet), testDat.Y);

    end
    
    E = [dnetLoss; isLoss; kjitLoss]';
    figure
    hold on 

    H = bar(E, 1, 'grouped');
    set(H(1), 'FaceColor', [0, 0, 0]);
    set(H(2), 'FaceColor', [0, 0, 0.8]);
    set(H(3), 'FaceColor', [0.8, 0, 0]);

    set(gca, 'FontSize', 18);
    legend('Infer.NET', 'Sampling', 'Sampling + KJIT' );
    set(gca, 'XTick', 1:length(dataLabels));
    set(gca, 'XTickLabel', dataLabels);


    ylabel('Error')
    %title(sprintf('Classification error on held-out test sets'));
    %legend('KJIT', 'Sampling', 'Infer.NET');
    legend('Infer.NET', 'Sampling', 'Sampling + KJIT');
    pbaspect([4 3 1]);
    grid on
    hold off

end


function plotTemporalUncertainty(kjit)
    % uncertaintyOutX over each time point 
    if nargin < 1
        kjit = getFileStructs();
    end
    xProblemInds = getProblemIndices(kjit);
    figure
    hold on 
    uncertaintyOutX = getAllUncertaintyOutX(kjit);
    % 1 for predicing the mean of X
    unSub = 1:min(7000, size(uncertaintyOutX, 2)); 
    window = 100;
    b = ones(1, window)/window;
    unOutXFil = filter(b, 1, uncertaintyOutX(1, :));
    xlim([1, max(unSub)]);
    ylim([-9.3, -8.3]);
    plot( uncertaintyOutX(1, unSub), '-b', 'LineWidth', 1);
    plot( unOutXFil(1, unSub), '-r', 'LineWidth', 2);
    % threshold 
    plot( unSub, -9*ones(1, length(unSub)), '-k', 'LineWidth', 1);

    % consultations 
    consultOracleOutX = getAllConsultations(kjit);
    consultTimes = find(consultOracleOutX(unSub));
    %plot(consultTimes, uncertaintyOutX(1, consultTimes), '^k');
    % draw vertical lines to indicate new problems

    [dataNames, dataLabels] = getDataNames();
    %dataLabelMap = containers.Map(dataNames, dataLabels);
    % draw problem labels
    for i=1:length(dataNames)
        %dn = dataNames{i};
        vline(xProblemInds(i), '-.*k', dataLabels{i});
    end
    set(gca, 'FontSize', 11);
    ylabel('Log predictive variance');
    xlabel('Factor invocations.');
    title('Predictive variance of the outgoing message')
    legend('Predictive variance', sprintf('Moving average'), 'Threshold');
    pbaspect([40, 5, 1]);
    %legend('Log predictive variance', sprintf('Moving average'), 'Consult oracle');
    hold off

end

function dnet = getInferNetStruct(kjit)
    % get struct results of Infer.NET 
    if nargin < 1 
        kjit = getFileStructs();
    end
    dataNames = fieldnames(kjit);
    dnet = struct();
    for i=1:length(dataNames)
        dn = dataNames{i};
        dnet.(dn).dnetPostMean = kjit.(dn).dnetPostMeans{1};
        dnet.(dn).dnetPostCov = kjit.(dn).dnetPostCovs{1};
    end
end

function [dataNames, dataLabels, testsetFNames] = getDataNames()
    % return the dataNames (file naming) and their dataLabels (for plotting).

    dataNames = {'banknote', 'blood', 'fertility', 'iono'};
    dataLabels = {'Banknote', 'Blood', 'Fertility', 'Ionosphere'};
    testsetFNames = {'banknote_norm_te.mat', 'blood_transfusion_norm_te.mat', ...
        'fertility_norm_te.mat', 'ionosphere_norm_te.mat'};
end

function fpathFunc = getFullFileFunc()
    scriptFol = Global.getScriptFolder();
    fpathFunc = @(fn)fullfile(scriptFol, 'logistic_msg', ...
        'infer.net_saved', 'online_uci', fn);
end

function [kjit, is] = getFileStructs()
    % Return a struct st containing fields identified by dataset names.
    % st.banknote, st.blood, ...

    % importance sampling size
    isSize = 100000;
    %isSize = 100000;
    epIter = 10;

    fullFileFunc = getFullFileFunc();
    dataNames = getDataNames();
    kjit = struct();
    is = struct();
    for i=1:length(dataNames)
        dn = dataNames{i};
        kjitFname = sprintf('kjit_is%d_%s_iter%d.mat', isSize, dn, epIter);
        kjitFpath = fullFileFunc(kjitFname);
        kjit.(dn) = load(kjitFpath);

        isFname = sprintf('is%d_%s_iter%d.mat', isSize, dn, epIter);
        isFpath = fullFileFunc(isFname);
        is.(dn) = load(isFpath);

        % load file example: 
        %    %is = 

        %     inLogOutLogTrue: [653x1 double]
        %    inLogOutLogFalse: [6233x1 double]
        %        inXOutLogMtp: [653x1 double]
        %       inXOutLogPrec: [653x1 double]
        %       inLogOutXTrue: [6233x1 double]
        %          inXOutXMtp: [6233x1 double]
        %         inXOutXPrec: [6233x1 double]
        %     outLogTrueCount: [653x1 double]
        %    outLogFalseCount: [653x1 double]
        %    projLogTrueCount: [653x1 double]
        %   projLogFalseCount: [653x1 double]
        %             outXMtp: [6233x1 double]
        %            outXPrec: [6233x1 double]
        %            projXMtp: [6233x1 double]
        %           projXPrec: [6233x1 double]
        % consultOracleOutLog: [653x1 double]
        %   uncertaintyOutLog: [2x653 double]
        %   consultOracleOutX: [6233x1 double]
        %     uncertaintyOutX: [2x6233 double]
        %  oraOutLogTrueCount: [653x1 double]
        % oraOutLogFalseCount: [653x1 double]
        % oraProjLogTrueCount: [653x1 double]
        %oraProjLogFalseCount: [653x1 double]
        %          oraOutXMtp: [6233x1 double]
        %         oraOutXPrec: [6233x1 double]
        %         oraProjXMtp: [6233x1 double]
        %        oraProjXPrec: [6233x1 double]
        %               extra: [1x1 struct]
        %      inferenceTimes: [4x1 double]
        %       dnetPostMeans: {4x1 cell}
        %        dnetPostCovs: {4x1 cell}
        %           postMeans: {4x1 cell}
        %            postCovs: {4x1 cell}
    end

end

function plotIncomingMessages(is)
    % plot incoming messages from the Gaussian variable in all problems.
    if nargin < 1
        [kjit, is] = getFileStructs();
    end
    [dataNames, dataLabels, testsetFNames] = getDataNames();
    figure 
    hold all
    styles = {'mh', 'k.', 'ro', 'bx'};
    for i=1:length(dataNames)
        dn = dataNames{i};
        mtp = is.(dn).inXOutXMtp;
        prec = is.(dn).inXOutXPrec;
        Means = mtp./prec;

        plot(Means, log(prec), styles{i});
    end
    set(gca, 'FontSize', 13);
    legend(dataLabels);
    ylabel('Log precision');
    xlabel('Mean');
    pbaspect([8, 3, 1]);
    hold off

end

function [kjitTimes, isTimes] = getInferenceTimes(kjit, is)
    if nargin < 2
        [kjit, is] = getFileStructs();
    end
    dataNames = fieldnames(kjit);
    kjitTimes = zeros(1, length(dataNames));
    isTimes = zeros(1, length(dataNames));
    for i=1:length(dataNames)
        dn = dataNames{i};
        kjitTimes(i) = kjit.(dn).inferenceTimes;
        isTimes(i) = is.(dn).inferenceTimes;
    end
end

function plotInferenceTimes(kjit, is)
    if nargin < 2
        [kjit, is] = getFileStructs();
    end
    [dataNames, dataLabels] = getDataNames();
    [kjitTimes, isTimes] = getInferenceTimes(kjit, is);
    figure 
    hold on 
    T = [isTimes(:), kjitTimes(:)];


    H = bar(T, 1, 'grouped');
    set(H(1), 'FaceColor', [0, 0, .8]);
    set(H(2), 'FaceColor', [.8, 0, 0]);
    set(gca, 'FontSize', 18);
    legend('Sampling', 'Sampling + KJIT' );
    %title('Inference time on real datasets.')
    ylabel('Time in ms');
    set(gca, 'XTick', 1:length(dataLabels));
    set(gca, 'XTickLabel', dataLabels);
    pbaspect([4 3 1]);

    grid on
    hold off 


end

function [kjit, is] = plotInferenceResults()
    [kjit, is] = getFileStructs();
    plotTemporalUncertainty(kjit);
    plotInferenceTimes(kjit, is);
    dnet = getInferNetStruct(kjit);
    plot01Loss(kjit, is, dnet);

    %plotPosteriorKL(kjit, is, dnet);
end

