function [ funcs] = funcs_logistic_uncertainty( )
%FUNCS_LOGISTIC_UNCERTAINTY Functions related to demonstrating uncertainty of 
% the operator on logistic data

    s = struct();

    s.plotUncertaintyAlongSlices = @plotUncertaintyAlongSlices;
    funcs = s;
end


function plotUncertaintyAlongSlices(s, trBundle, teBundle, subsample)
    % Plot uncertainty along a slice (line) passing from unexplored region, 
    % through a dense region of the training set, and out to unexplored region.
    % Expect a U-shaped curve of uncertainty.
    %
    % Input: 
    % - s = struct of learned operator. Example:
    % s = 
    %
    %  learner_class: 'RFGJointKGGLearner'
    %learner_options: [1x1 Options]
    %    dist_mapper: [1x1 UAwareGenericMapper]
    %    learner_log: [1x1 struct]
    %         commit: '0dac41dc'
    %      timeStamp: [2015 2 18 17 54 12.6150]
    %

    oldRng = rng();
    rng(1);

    if nargin < 4
        subsample = 3000;
    end
    assert(isa(s.dist_mapper, 'UAwareDistMapper'), 's.dist_mapper must be UAwareDistMapper');
    trIns = trBundle.getInputBundles();
    trLogistic = trIns{1};
    trX = trIns{2};
    trX_mean = [trX.mean];
    trX_var = [trX.variance];

    teIns = teBundle.getInputBundles();
    teLogistic = teIns{1};
    teX = teIns{2};
    teX_mean = [teX.mean];
    teX_var = [teX.variance];

    % The previous line is to remove messages possibly resulted from Infer.NET's 
    % truncation i.e., SetToRatio(..) with force proper.
    Itr =  ~(abs(trX_mean) <= 1e-3 & log(trX_var)>0 );
    Ite =  ~(abs(teX_mean) <= 1e-3 & log(teX_var)>0 );
    trX_mean_fil = trX_mean(Itr);
    teX_mean_fil = teX_mean(Ite);
    trX_var_fil = trX_var(Itr);
    teX_var_fil = teX_var(Ite);

    % subsampling for plotting purpose 
    ntr = length(trX_mean_fil);
    nte = length(teX_mean_fil);
    subTr = randperm(ntr, min(subsample, ntr));
    subTe = randperm(nte, min(subsample, nte));
    trX_mean_fil = trX_mean_fil(subTr);
    teX_mean_fil = teX_mean_fil(subTe);
    trX_var_fil = trX_var_fil(subTr);
    teX_var_fil = teX_var_fil(subTe);
    
    % line passing through the cloud
    % A function mapping means to log(precision)
    testMeans = linspace(-10, 10);
    strLine = @(m)2e-2*m + 2.6;
    paraLine = @(m)-2e-2*m.^2 + 1.6;
    str2Line = @(m)-12e-2*m + 1.5;

    strPrecs = strLine(testMeans);
    paraPrecs = paraLine(testMeans);
    str2Precs = str2Line(testMeans);

    betaMsg = DistBeta(1, 2);
    [betaMsgs, strNormalMsgs] = genMsgBundleFromMeanPrec(testMeans, strPrecs, betaMsg);
    U_straight = s.dist_mapper.estimateUDistArrays(betaMsgs, strNormalMsgs );

    [betaMsgs, paraNormalMsgs] = genMsgBundleFromMeanPrec(testMeans, paraPrecs, betaMsg);
    U_para = s.dist_mapper.estimateUDistArrays(betaMsgs, paraNormalMsgs);

    [betaMsgs, str2NormalMsgs] = genMsgBundleFromMeanPrec(testMeans, str2Precs, betaMsg);
    U_str2 = s.dist_mapper.estimateUDistArrays(betaMsgs, str2NormalMsgs);

    xlabel_text ='Mean of incoming Gaussian messages'; 
    %
    % plot  data points
    fontsize = 15;
    figure 
    subplot(1, 2, 1);
    hold on 
    %plot(teX_mean_fil, -log(teX_var_fil), 'xr', 'LineWidth', 1);
    plot(trX_mean_fil, -log(trX_var_fil), '*k', 'LineWidth', 1, 'MarkerSize', 4 );
    plot(testMeans, strPrecs, '-b', 'LineWidth', 2)
    plot(testMeans, paraPrecs, '-m', 'LineWidth', 2)
    plot(testMeans, str2Precs, '-r', 'LineWidth', 2)
    set(gca, 'FontSize', fontsize);
    %legend('Test set', 'Training set');
    legend('Training set', 'Uncertainty test #1', 'Uncertainty test #2', ...
        'Uncertainty test #3');
    xlabel(xlabel_text);
    ylabel('Log precision');
    xlim([min(testMeans)-1, max(testMeans)+1]);
    hold off 

    % plot uncertainty 
    %figure 
    subplot(1, 2, 2);
    hold on 
    plot(testMeans, log(U_straight(1, :)), '-b', 'LineWidth', 2);
    plot(testMeans, log(U_straight(2, :)), '--b', 'LineWidth', 2);
    plot(testMeans, log(U_para(1, :)), '-m', 'LineWidth', 2);
    plot(testMeans, log(U_para(2, :)), '--m', 'LineWidth', 2);
    plot(testMeans, log(U_str2(1, :)), '-r', 'LineWidth', 2);
    plot(testMeans, log(U_str2(2, :)), '--r', 'LineWidth', 2);
    set(gca, 'FontSize', fontsize);
    superTitle=sprintf('%s', 'Log predictive variance on the uncertainty test sets. m_{z\rightarrow f}=Beta(1, 2).');
    annotation('textbox', [0 0.9 1 0.1], ...
        'String', superTitle, ...
        'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 18, 'interpreter', 'tex')
    legend('#1: Predict mean', '#1: Predict log variance', ...
        '#2: Predict mean', '#2: Predict log variance', ...
        '#3: Predict mean', '#3: Predict log variance' ...
        );
    xlabel(xlabel_text);
    ylabel('Log of predictive variance')
    xlim([min(testMeans)-1, max(testMeans)+1]);
    hold off 

    rng(oldRng);
end

function [betaMsgs, normalMsgs] = genMsgBundleFromMeanPrec(means, logPrecs, betaMsg)
    assert(isa(betaMsg, 'DistBeta'));
    n = length(means);
    vars = exp(-logPrecs);
    normalMsgs = DistNormal(means, vars);
    betaMsgs = DistBeta(repmat(betaMsg.alpha, 1, n), repmat(betaMsg.beta, 1, n));
end
