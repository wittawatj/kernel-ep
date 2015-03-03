function [ funcs] = funcs_logistic_uncertainty( )
%FUNCS_LOGISTIC_UNCERTAINTY Functions related to demonstrating uncertainty of 
% the operator on logistic data

    s = struct();

    s.plotUncertaintyAlongSlices = @plotUncertaintyAlongSlices;
    s.gen2DUncertaintyCheckData = @gen2DUncertaintyCheckData;
    s.gen2DUncertaintyCheckData2 = @gen2DUncertaintyCheckData2;
    s.plotUncertaintyAlongSlicesSuperImposed = @plotUncertaintyAlongSlicesSuperImposed;
    funcs = s;
end

function plotUncertaintyAlongSlicesSuperImposed(s, trBundle, teBundle, Un )
    % Same as the other function with similar name but superimpose other method's
    % Uncertainty on top.

    oldRng = rng();
    rng(1);

    subsample = 2000;
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

    % remove messages possibly resulted from Infer.NET's 
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
    testMeans = linspace(-13, 13, 500);
    strLine = @(m)2e-2*m + 2.6;
    paraLine = @(m)-2e-2*m.^2 + 1.6;
    str2Line = @(m)-12e-2*m + 1.5;

    strLogPrecs = strLine(testMeans);
    paraLogPrecs = paraLine(testMeans);
    str2LogPrecs = str2Line(testMeans);

    betaMsg = DistBeta(1, 2);
    [betaMsgs, strNormalMsgs] = genMsgBundleFromMeanLogPrec(testMeans, strLogPrecs, betaMsg);
    U_straight = s.dist_mapper.estimateUDistArrays(betaMsgs, strNormalMsgs );

    [betaMsgs, paraNormalMsgs] = genMsgBundleFromMeanLogPrec(testMeans, paraLogPrecs, betaMsg);
    U_para = s.dist_mapper.estimateUDistArrays(betaMsgs, paraNormalMsgs);

    %[betaMsgs, str2NormalMsgs] = genMsgBundleFromMeanLogPrec(testMeans, str2LogPrecs, betaMsg);
    %U_str2 = s.dist_mapper.estimateUDistArrays(betaMsgs, str2NormalMsgs);

    %xlabel_text ='Mean of incoming Gaussian messages'; 
    xlabel_text ='Mean'; 
    %
    % plot  data points
    fontsize = 16;
    figure 
    %subplot(1, 2, 1);
    hold on 
    %plot(teX_mean_fil, -log(teX_var_fil), 'xr', 'LineWidth', 1);
    plot(trX_mean_fil, -log(trX_var_fil), '*k', 'LineWidth', 1, 'MarkerSize', 4 );
    plot(testMeans, strLogPrecs, '-b', 'LineWidth', 2)
    plot(testMeans, paraLogPrecs, '-m', 'LineWidth', 2)
    %plot(testMeans, str2LogPrecs, '-r', 'LineWidth', 2)
    set(gca, 'FontSize', fontsize);
    %legend('Test set', 'Training set');
    %legend('Training set', 'Uncertainty test #1', 'Uncertainty test #2', ...
    %    'Uncertainty test #3');
    legend('Training set', 'Uncertainty test #1', 'Uncertainty test #2');
    xlabel(xlabel_text);
    ylabel('Log precision');
    xlim([min(testMeans)-1, max(testMeans)+1]);
    pbaspect([4 3 1]);
    hold off 

    % plot uncertainty 
    figure 
    %subplot(1, 2, 2);
    hold on 

    % other methods
    plot(testMeans, Un(1, :), '--b');
    plot(testMeans, Un(2, :), '--m');

    % out kernel-based method
    plot(testMeans, log(U_straight(1, :)), '-b', 'LineWidth', 2);
    %plot(testMeans, log(U_straight(2, :)), '--b', 'LineWidth', 2);
    plot(testMeans, log(U_para(1, :)), '-m', 'LineWidth', 2);
    %plot(testMeans, log(U_para(2, :)), '--m', 'LineWidth', 2);
    %plot(testMeans, log(U_str2(1, :)), '-r', 'LineWidth', 2);
    %plot(testMeans, log(U_str2(2, :)), '--r', 'LineWidth', 2);
    set(gca, 'FontSize', fontsize);
    %legend('Uncertainty test #1', 'Uncertainty test #2');
    legend('Random forests: #1', 'Random forests: #2', ...
        'Kernel: #1', 'Kernel: #2');
    xlabel(xlabel_text);
    ylabel('Log predictive variance')
    xlim([min(testMeans)-1, max(testMeans)+1]);
    pbaspect([4 3 1]);
    grid on;
    hold off 

    rng(oldRng);
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
        subsample = 2000;
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

    % remove messages possibly resulted from Infer.NET's 
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
    testMeans = linspace(-13, 13, 500);
    strLine = @(m)2e-2*m + 2.6;
    paraLine = @(m)-2e-2*m.^2 + 1.6;
    str2Line = @(m)-12e-2*m + 1.5;

    strLogPrecs = strLine(testMeans);
    paraLogPrecs = paraLine(testMeans);
    str2LogPrecs = str2Line(testMeans);

    betaMsg = DistBeta(1, 2);
    [betaMsgs, strNormalMsgs] = genMsgBundleFromMeanLogPrec(testMeans, strLogPrecs, betaMsg);
    U_straight = s.dist_mapper.estimateUDistArrays(betaMsgs, strNormalMsgs );

    [betaMsgs, paraNormalMsgs] = genMsgBundleFromMeanLogPrec(testMeans, paraLogPrecs, betaMsg);
    U_para = s.dist_mapper.estimateUDistArrays(betaMsgs, paraNormalMsgs);

    %[betaMsgs, str2NormalMsgs] = genMsgBundleFromMeanLogPrec(testMeans, str2LogPrecs, betaMsg);
    %U_str2 = s.dist_mapper.estimateUDistArrays(betaMsgs, str2NormalMsgs);

    %xlabel_text ='Mean of incoming Gaussian messages'; 
    xlabel_text ='Mean'; 
    %
    % plot  data points
    fontsize = 16;
    figure 
    %subplot(1, 2, 1);
    hold on 
    %plot(teX_mean_fil, -log(teX_var_fil), 'xr', 'LineWidth', 1);
    plot(trX_mean_fil, -log(trX_var_fil), '*k', 'LineWidth', 1, 'MarkerSize', 4 );
    plot(testMeans, strLogPrecs, '-b', 'LineWidth', 2)
    plot(testMeans, paraLogPrecs, '-m', 'LineWidth', 2)
    %plot(testMeans, str2LogPrecs, '-r', 'LineWidth', 2)
    set(gca, 'FontSize', fontsize);
    %legend('Test set', 'Training set');
    %legend('Training set', 'Uncertainty test #1', 'Uncertainty test #2', ...
    %    'Uncertainty test #3');
    legend('Training set', 'Uncertainty test #1', 'Uncertainty test #2');
    xlabel(xlabel_text);
    ylabel('Log precision');
    xlim([min(testMeans)-1, max(testMeans)+1]);
    pbaspect([4 3 1]);
    hold off 

    % plot uncertainty 
    figure 
    %subplot(1, 2, 2);
    hold on 
    plot(testMeans, log(U_straight(1, :)), '-b', 'LineWidth', 2);
    %plot(testMeans, log(U_straight(2, :)), '--b', 'LineWidth', 2);
    plot(testMeans, log(U_para(1, :)), '-m', 'LineWidth', 2);
    %plot(testMeans, log(U_para(2, :)), '--m', 'LineWidth', 2);
    %plot(testMeans, log(U_str2(1, :)), '-r', 'LineWidth', 2);
    %plot(testMeans, log(U_str2(2, :)), '--r', 'LineWidth', 2);
    set(gca, 'FontSize', fontsize);
    %superTitle=sprintf('%s', ['Log predictive variance on the uncertainty test sets. ' , ...
    %'Fix m_{z\rightarrow f}=Beta(1, 2) in the test sets.']);
    %
    %superTitle=sprintf('%s', 'Log predictive variance on the uncertainty test sets.');
    %annotation('textbox', [0 0.9 1 0.1], ...
    %    'String', superTitle, ...
    %    'EdgeColor', 'none', ...
    %    'HorizontalAlignment', 'center', ...
    %    'FontSize', 18, 'interpreter', 'tex')
    %
    %legend('#1: Predict mean', '#1: Predict log variance', ...
    %    '#2: Predict mean', '#2: Predict log variance', ...
    %    '#3: Predict mean', '#3: Predict log variance' ...
    %    );
    %legend('#1: Predict mean', '#1: Predict log variance', ...
    %   '#2: Predict mean', '#2: Predict log variance' );
    legend('Uncertainty test #1', 'Uncertainty test #2');
    xlabel(xlabel_text);
    ylabel('Log of predictive variance')
    xlim([min(testMeans)-1, max(testMeans)+1]);
    pbaspect([4 3 1]);
    grid on;
    hold off 

    rng(oldRng);
end

function [betaMsgs, normalMsgs] = genMsgBundleFromMeanLogPrec(means, logPrecs, betaMsg)
    assert(isa(betaMsg, 'DistBeta'));
    n = length(means);
    vars = exp(-logPrecs);
    normalMsgs = DistNormal(means, vars);
    betaMsgs = DistBeta(repmat(betaMsg.alpha, 1, n), repmat(betaMsg.beta, 1, n));
end

function [Xtr, Ytr, Xte1, Xte2, Ttr, Tte1, Tte2] = gen2DUncertaintyCheckData2(trBundle )
    % This method is used to generate data for Balaji.
    % Include tree parameterization of Ali et. al (see appendix).
    %
    % input s:
    % s = 
    %  learner_class: 'RFGJointKGGLearner'
    %learner_options: [1x1 Options]
    %    dist_mapper: [1x1 UAwareGenericMapper]
    %    learner_log: [1x1 struct]
    %         commit: '0dac41dc'
    %      timeStamp: [2015 2 18 17 54 12.6150]

    trIns = trBundle.getInputBundles();
    trLog = trIns{1}.distArray;
    trLog_alpha = [trLog.alpha];
    trLog_beta = [trLog.beta];

    trX = trIns{2};
    trX_mean = [trX.mean];
    trX_var = [trX.variance];

    % remove messages possibly resulted from Infer.NET's 
    % truncation i.e., SetToRatio(..) with force proper.
    Itruncate =  ~(abs(trX_mean) <= 1e-3 & log(trX_var)>0 );
    Itr = Itruncate ;
    %Ite =  ~(abs(teX_mean) <= 1e-3 & log(teX_var)>0 );
    trX_mean_fil = trX_mean(Itr);
    trX_var_fil = trX_var(Itr);
    trLog_alpha_fil = trLog_alpha(Itr);
    trLog_beta_fil = trLog_beta(Itr);

    ntr = length(trX_mean_fil);
    Xtr(:, 1) = trLog_alpha_fil(:);
    Xtr(:, 2) = trLog_beta_fil(:);
    Xtr(:, 3) = trX_mean_fil(:);
    % log precision
    Xtr(:, 4) = -log(trX_var_fil(:));
    
    % mean of Xtr 
    daOut = trBundle.getOutBundle();
    Ytr = [ [daOut.mean] ; -log([daOut.variance]) ]';
    Ytr = Ytr(Itr, :);

    nte = 500;
    testMeans = linspace(-10, 10, nte);
    strLine = @(m)2e-2*m + 2.6;
    paraLine = @(m)-2e-2*m.^2 + 1.6;

    strLogPrecs = strLine(testMeans);
    paraLogPrecs = paraLine(testMeans);

    % beta message fixed to (alpha=1, beta=2) in the test sets
    Xte1(:, 1:2) = repmat([1, 2], nte, 1 );
    Xte1(:, 3) = testMeans(:);
    Xte1(:, 4) = strLogPrecs(:);
    Xte2(:, 1:2) = Xte1(:, 1:2);
    Xte2(:, 3) = testMeans(:);
    Xte2(: ,4) = paraLogPrecs(:);

    logistic = @(x)(1./(1+exp(-x)));
    % Tree parameterization
    Ttr = [treeParameterizeBeta(trLog(Itr), logistic), treeParameterizeGauss(trX(Itr), logistic) ];
    te1Log = DistBeta(ones(1, nte), 2*ones(1, nte));
    te1X = DistNormal(testMeans(:)', 1./exp(strLogPrecs(:)') );
    Tte1 = [treeParameterizeBeta(te1Log, logistic), treeParameterizeGauss(te1X, logistic) ];

    te2Log = te1Log;
    te2X = DistNormal(testMeans(:)', 1./exp(paraLogPrecs(:)') );
    Tte2 = [treeParameterizeBeta(te2Log, logistic), treeParameterizeGauss(te2X, logistic) ];

end

function T = treeParameterizeGauss(da, fac)
    % fac is a function handle representing the factor
    % Expand the distribution array into features as described in Ali et. al in the 
    % appendix.
    assert(isa(da, 'DistNormal'));

    prec = 1./[da.variance];
    means = [da.mean];
    % mean = mode. Include the values of the factor evaluated at the modes.
    X = [ means; [da.variance]; means.*prec; prec; fac(means)];
    T = X';
end

function T = treeParameterizeBeta(da, fac)
    % fac is a function handle representing the factor
    % Expand the distribution array into features as described in Ali et. al in the 
    % appendix.
    assert(isa(da, 'DistBeta'));
    a = [da.alpha];
    b = [da.beta];
    means = [da.mean];
    vars =[da.variance];
    % modes do not exist for a<1 or b<1 ?
    modes = (a-1)./(a + b-2);
    X = [a; b; means; vars; fac(modes)];
    T = X';
end

function [X, Y ] = gen2DUncertaintyCheckData(st, subsample)
    % generate a 2-dimensional simple regression dataset from the struct loaded 
    % from a file containing all messages collected from running EP in Infer.NET 
    % Expected st :
    %
    %st = 
    %        outNormalMeans: [40000x1 double]
    %    outNormalVariances: [40000x1 double]
    %         inNormalMeans: [40000x1 double]
    %     inNormalVariances: [40000x1 double]
    %               inBetaA: [40000x1 double]
    %               inBetaB: [40000x1 double]
    %
    % - Inputs are (inNormalMeans, log(inNormalVariances)) corresponding to 
    % Beta(1, 2) incoming messages.
    % - Output = outNormalMeans.
    %
    assert(subsample > 0);
    Ibeta = abs(st.inBetaA-1) <=1e-8 & abs(st.inBetaB-2) <= 1e-8;
    I = Ibeta & st.inNormalMeans <= -1.0;
    Iunseen = Ibeta & st.inNormalMeans >= 1.0;
    
    n = sum(I);
    nunseen = sum(Iunseen);
    X = zeros(n, 2);
    X(:, 1) = st.inNormalMeans(I);
    X(:, 2) = log(st.inNormalVariances(I));
    Y = st.outNormalMeans(I);

    Xuns = zeros(nunseen, 2);
    Xuns(:, 1) = st.inNormalMeans(Iunseen);
    Xuns(:, 2) = log(st.inNormalVariances(Iunseen));
    Yuns = st.outNormalMeans(Iunseen);

    I_sub = randperm(n, min(n, subsample));
    Iuns_sub = randperm(nunseen, min(nunseen, subsample));
    X = X(I_sub, :);
    Y = Y(I_sub);

    Xuns = Xuns(Iuns_sub, :);
    Yuns = Yuns(Iuns_sub);

end


