function [ s] = funcs_logistic_online( )
%FUNCS_LOGISTIC_ONLINE Functions related to online learning with a logistic message 
%operator 

    s = struct();

    s.plotInferenceResults = @plotInferenceResults;
    s.plotIncomingMsgs = @plotIncomingMsgs;
end

function plotIncomingMsgs(st)
    % st is a struct containing all loaded variables.
    figure;
    inMean = st.inXOutXMtp./st.inXOutXPrec;
    inVariance = 1./st.inXOutXPrec;
    plot(inMean, inVariance, 'o');
    set(gca, 'FontSize', 14);
    xlabel('incoming mean');
    ylabel('incoming variance');
    title('Sending to x');

    figure;
    hist(log(inVariance));
    set(gca, 'FontSize', 14);
    title('incoming log variance when sending to x');

    
end

function kepInferTimes = getKepInferTimes(kepCells)

    kepInferTimes = zeros(1, length(kepCells));
    for i=1:length(kepCells)
        kep = kepCells{i};
        kepInferTimes(i) = kep.inferenceTimes;
    end
end

function st = getExtraStructs(kepCells)
    % get the extra structs in kepCells 
    % Return a struct array
    stCell = cell(1, length(kepCells));
    for i=1:length(kepCells)
        stCell{i} = kepCells{i}.extra;
    end
    st = [stCell{:}];

end

function [kepLoss, isLoss, dnetLoss] = plot01Loss(kepCells, isCells, dnetPost)
    % generate some test data from true W, use posterior mean as the decision 
    % function. Plot 0-1 loss.

    assert(isa(dnetPost, 'DistNormal'));
    [kepPost] = getPosteriors(kepCells);
    [isPost] = getPosteriors(isCells);
    assert(isa(kepPost, 'DistNormal'));
    % extras are the same for kepCells and isCells containing infor. of logistic 
    % problems.
    extras = getExtraStructs(kepCells);

    %extra = 
    %         d: 20
    %         n: 400
    %    epIter: 10
    %     trueW: [20x1 double]
    %         X: [20x400 double]
    %         Y: [400x1 double]

    % test set 
    nte = 10000;
    trials = length(kepPost);
    sigFunc = @(p)1./(1+ exp(-p));
    
    kepLoss = zeros(1, trials);
    isLoss = zeros(1, trials);
    dnetLoss = zeros(1, trials);
    for i=1:trials
        d = extras(i).d;
        trueW = extras(i).trueW;
        Xte = randn(d, nte);
        Pte = sigFunc( trueW(:)'*Xte);
        Yte = rand(1, nte) <= Pte;

        % test with posterior means
        kepMean = kepPost(i).mean;
        Pkep = sigFunc(kepMean(:)'*Xte);
        % average classification error (0-1 loss)
        kepLoss(i) = mean(round(Pkep) ~= round(Pte));

        isMean = isPost(i).mean;
        Pis = sigFunc(isMean(:)'*Xte);
        isLoss(i) = mean(round(Pis) ~= round(Pte));

        dnetMean = dnetPost(i).mean;
        Pdnet = sigFunc(dnetMean(:)'*Xte);
        dnetLoss(i) = mean(round(Pdnet) ~= round(Pte));
    end
    
    figure
    hold on 
    T = 1:trials;
    plot(T, dnetLoss, '-k+', 'LineWidth', 2);
    plot(T, isLoss, '-bd', 'LineWidth', 2);
    plot(T, kepLoss, '-r', 'LineWidth', 2);
    set(gca, 'FontSize', 18);
    xlabel('Problems seen');
    ylabel('Classification error')
    title(sprintf('Test size: %d.', nte ));
    %legend('KJIT', 'Sampling', 'Infer.NET');
    legend('Infer.NET', 'Sampling', 'Sampling + KJIT');
    grid on
    hold off


end

function [kepPost] = getPosteriors(kepCells)
    % also work for isCells
    kepPostMeans = [];
    kepPostCovs = [];
    for i=1:length(kepCells)
        kep = kepCells{i};
        km = kep.postMeans{1};
        kc = kep.postCovs{1};
        kepPostMeans = [kepPostMeans, km];
        kepPostCovs = cat(3, kepPostCovs, kc );

    end
    kepPost = DistNormal(kepPostMeans, kepPostCovs);
end

function plotPosteriorKL(kepCells, isCells)
    % plot agreement of posteriors between sampling vs. Infer.NET.
    % KJIT vs sampling.

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
    plot(seeds, log(klIsKep), '-*r', 'LineWidth', 2);
    plot(seeds, log(klDnetKep), '-db', 'LineWidth', 2);
    plot(seeds, log(klDnetIs), '-+k', 'LineWidth', 2);
    set(gca, 'FontSize', 18);
    xlabel('Problems seen');
    ylabel('Log KL divergence')
    title('KL divergence of the posteriors')
    legend('KL(sampling, KJIT)', 'KL(Infer.NET, KJIT)', 'KL(Infer.NET, sampling)');
    grid on
    hold off


end

function [kepCells, isCells, kepInferTimes, isInferTimes, ...
        dnetInferTimes, xProblemInds, uncertaintyOutX]= getFileStructs()
    % get loaded file structs
    
    % plot inference time and other results of the online logistic factor problem.
    %
  
    %kepFilePrefix = sprintf('rec_onlinekep_is%d_logistic_iter%d_s*', isSize, epIter);
    %glob = fullfile(scriptFol, 'logistic_msg', 'infer.net_saved', kepFilePrefix);
    %files = dir(glob);
    %% sorted files based on names (numerical order)
    %soFiles = struct();
    %for i=1:length(files)
    %    f = files(i);
    %    match = regexp(f.name, 'rec_onlinekep_is(?<seed>\d+)_logistic_iter(?<iter>\d+).mat', 'names');
    %    seed = match.seed; 
    %end

    % importance sampling size
    isSize = 20000;
    %isSize = 100000;
    epIter = 10;
    n = 300;
    seed_to = 2;
    seeds =  1:seed_to;
    scriptFol = Global.getScriptFolder();
    fullFileFunc = @(fn)fullfile(scriptFol, 'logistic_msg', ...
        'infer.net_saved',  fn);
    kepFNames =arrayfun(@(s)sprintf('rec_onlinekep_is%d_n%d_logistic_iter%d_s%d.mat', isSize, n, epIter, s), ...
        seeds, 'UniformOutput', false);
    kepFPaths = cellfun(fullFileFunc, kepFNames, 'UniformOutput', false );
    isFNames =arrayfun(@(s)sprintf('rec_is%d_n%d_logistic_iter%d_s%d.mat', isSize, n, epIter, s), ...
        seeds, 'UniformOutput', false);
    isFPaths = cellfun(fullFileFunc, isFNames, 'UniformOutput', false );

    kepInferTimes = zeros(1, length(seeds));
    isInferTimes = zeros(1, length(seeds));
    uncertaintyOutX = [];
    % indices of new problems based on the count of X messages
    xProblemInds = [];
    kepCells = cell(1, length(seeds));
    isCells = cell(1, length(seeds));


    % Example a loaded file
    %consultOracleOutX         4924x1               39392  double              
    %dnetPostCovs                 1x1                3312  cell                
    %dnetPostMeans                1x1                 272  cell                
    %extra                        1x1               85240  struct              
    %inLogOutLogFalse          4924x1               39392  double              
    %inLogOutLogTrue            500x1                4000  double              
    %inLogOutXTrue             4924x1               39392  double              
    %inXOutLogMtp               500x1                4000  double              
    %inXOutLogPrec              500x1                4000  double              
    %inXOutXMtp                4924x1               39392  double              
    %inXOutXPrec               4924x1               39392  double              
    %inferenceTimes               1x1                   8  double              
    %oraOutLogFalseCount        500x1                4000  double              
    %oraOutLogTrueCount         500x1                4000  double              
    %oraOutXMtp                4924x1               39392  double              
    %oraOutXPrec               4924x1               39392  double              
    %oraProjLogFalseCount       500x1                4000  double              
    %oraProjLogTrueCount        500x1                4000  double              
    %oraProjXMtp               4924x1               39392  double              
    %oraProjXPrec              4924x1               39392  double              
    %outLogFalseCount           500x1                4000  double              
    %outLogTrueCount            500x1                4000  double              
    %outXMtp                   4924x1               39392  double              
    %outXPrec                  4924x1               39392  double              
    %postCovs                     1x1                3312  cell                
    %postMeans                    1x1                 272  cell                
    %projLogFalseCount          500x1                4000  double              
    %projLogTrueCount           500x1                4000  double              
    %projXMtp                  4924x1               39392  double              
    %projXPrec                 4924x1               39392  double              
    %uncertaintyOutLog            2x500              8000  double              
    %uncertaintyOutX              2x4924            78784  double  
    prevProblemI = 1;
    for i=1:length(seeds)
        display(sprintf('loading: %s', kepFPaths{i}));
        kep = load(kepFPaths{i});
        kepCells{i} = kep;
        kepInferTimes(i) = kep.inferenceTimes;
        uncertaintyOutX = [uncertaintyOutX, kep.uncertaintyOutX];
        xProblemInds(end+1) = prevProblemI;
        prevProblemI = prevProblemI + length(kep.inXOutXPrec);

        display(sprintf('loading: %s', isFPaths{i}));
        is = load(isFPaths{i});
        isInferTimes(i) = is.inferenceTimes;
        isCells{i} = is;

    end

    % Infer.net 
    dnet = loadDnetResults();
    dnetInferTimes = dnet.allInferTimes;
end

function dnet = loadDnetResults()

    scriptFol = Global.getScriptFolder();
    dnetSeedTo = 100;
    fn = sprintf('rec_dnet_logistic_iter10_sf1_st%d.mat', dnetSeedTo);
    dnetPath = fullfile(scriptFol, 'logistic_msg', 'infer.net_saved', fn);
    dnet = load(dnetPath);
end

function plotInferenceResults()

    [kepCells, isCells, kepInferTimes, isInferTimes, ...
    dnetInferTimes, xProblemInds, uncertaintyOutX]= getFileStructs();
    seed_to = length(kepCells);
    seeds = 1:seed_to;
    % plot inference time.
    timeSub = 1:min(20, length(seeds));
    figure
    hold on 
    plot(timeSub, log(dnetInferTimes(timeSub)), '+-k', 'LineWidth', 2);
    plot(timeSub, log(isInferTimes(timeSub)), '-db', 'LineWidth', 2);
    plot(timeSub, log(kepInferTimes(timeSub)), '-r', 'LineWidth', 2);
    set(gca, 'FontSize', 18);
    ylabel('Time in log(ms)')
    xlabel('Problems seen');
    title('Inference time')
    legend('Infer.NET', 'Sampling', 'Sampling + KJIT');
    grid on
    hold off

    % uncertaintyOutX over each time point 
    figure
    hold on 
    % 1 for predicing the mean of X
    unSub = 1:min(2500, size(uncertaintyOutX, 2)); 
    window = 60;
    b = ones(1, window)/window;
    unOutXFil = filter(b, 1, uncertaintyOutX(1, :));
    plot( uncertaintyOutX(1, unSub), '-b', 'LineWidth', 1);
    plot( unOutXFil(1, unSub), '-r', 'LineWidth', 2);
    % draw vertical lines to indicate new problems
    vline(xProblemInds, '-.*k');
    xlim([1, max(unSub)]);
    set(gca, 'FontSize', 12);
    ylabel('Log of predictive variance');
    xlabel('Time for each input');
    title('Predictive variance of incoming messages at each time point')
    legend('Log predictive variance', sprintf('Moving average'));
    hold off

    plotPosteriorKL(kepCells, isCells);

    % load Infer.NET results
    dnet = loadDnetResults();
    dnetMeans = [dnet.postMeans{:}];
    dnetCovs = cat(3, dnet.postCovs{:});
    dnetPost = DistNormal(dnetMeans, dnetCovs);
    [kepLoss, isLoss, dnetLoss] = plot01Loss(kepCells, isCells, dnetPost);
end
