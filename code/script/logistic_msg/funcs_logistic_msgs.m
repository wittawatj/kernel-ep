function [ s ] = funcs_logistic_msgs(  )
%FUNCS_LOGISTIC_MSGS A set of functions to implement necessary operations to 
%bring object from/to Infer.NET
%

    % Return a struct containing functions for exp3
    s=struct();
    s.convertBinLogFWBundle = @convertBinLogFWBundle;
    s.convertBinLogBWBundle = @convertBinLogBWBundle;
    s.genBinLogFWBundle = @genBinLogFWBundle;
    s.genBinLogBWBundle = @genBinLogBWBundle;
    s.genSigmoidFactorOperator = @genSigmoidFactorOperator;
    s.getSigmoidFactorOperator = @getSigmoidFactorOperator;
    s.gen2DUncertaintyCheckData = @gen2DUncertaintyCheckData;
    s.gen2DSimpleRegression = @gen2DSimpleRegression;
end

function bundle = convertBinLogFWBundle(bundleName)
    % Read, convert and write to saved/bundle/ folder.
    scriptFol = Global.getScriptFolder();
    fromInferFile = fullfile(scriptFol, 'logistic_msg', 'infer.net_saved', bundleName);
    s = load(fromInferFile);
    bundle = genBinLogFWBundle(s, bundleName);
end

function bundle=genBinLogFWBundle(s, bundleName )
% Generate a MsgBundle from an exported .mat file from 
%Infer.NET containing recorded incoming/outgoing messages during EP inference 
%on a binary logistic regression model. Forward direction i.e. to Beta variable.
%
%  Input:
%  - s is the struct obtained by s = load('...')
%
% Example of what is inside the .mat file 
%
%%  Name                            Size            Bytes  Class      Attributes
%  X                             400x1             76800  cell                 
%  Y                             400x1               400  logical              
%  inBetaA                       400x1              3200  double               
%  inBetaB                       400x1              3200  double               
%  inNormalMeans                 400x1              3200  double               
%  inNormalVariances             400x1              3200  double               
%  outBetaA                      400x1              3200  double               
%  outBetaB                      400x1              3200  double               
%  regression_input_dim            1x1                 8  double               
%  regression_training_size        1x1                 8  double               
%  true_bias                       1x1                 8  double               
%  true_w                         10x1                80  double               


isProperFunc = @(d)d.isProper();
inBetas = DistBeta(s.inBetaA', s.inBetaB');
inBetasProper = arrayfun(isProperFunc, inBetas);

inNormals = DistNormal(s.inNormalMeans', s.inNormalVariances');
inNormalsProper = arrayfun(isProperFunc, inNormals);

outBetas = DistBeta(s.outBetaA', s.outBetaB');
outBetasProper = arrayfun(isProperFunc, outBetas);

% indices of proper messages
I = inBetasProper & inNormalsProper & outBetasProper;

inBetas = DistArray(inBetas(I));
inNormals = DistArray(inNormals(I));
outBetas = DistArray(outBetas(I));

bundle = DefaultMsgBundle(outBetas, inBetas, inNormals);
serializer = BundleSerializer();
serializer.saveBundle(bundle, bundleName);

end

function bundle = convertBinLogBWBundle(bundleName)
    % Read, convert and write to saved/bundle/ folder.
    scriptFol = Global.getScriptFolder();
    fromInferFile = fullfile(scriptFol, 'logistic_msg', 'infer.net_saved', bundleName);
    s = load(fromInferFile);
    bundle = genBinLogBWBundle(s, bundleName);
end

function bundle=genBinLogBWBundle(s, bundleName )
% Generate a MsgBundle from an exported .mat file from 
%Infer.NET containing recorded incoming/outgoing messages during EP inference 
%on a binary logistic regression model. Backward direction i.e. to Gaussian 
% variable.
%
%  Input:
%  - s is the struct obtained by s = load('...')
%
% Example of what is inside the .mat file 
%
%  X                              300x1               57600  cell                 
%  Y                              300x1                 300  logical              
%  ans                              1x6300            50400  double               
%  inBetaA                       6300x1               50400  double               
%  inBetaB                       6300x1               50400  double               
%  inNormalMeans                 6300x1               50400  double               
%  inNormalVariances             6300x1               50400  double               
%  outNormalMeans                6300x1               50400  double               
%  outNormalVariances            6300x1               50400  double               
%  regression_input_dim             1x1                   8  double               
%  regression_training_size         1x1                   8  double               
%  true_bias                        1x1                   8  double               
%  true_w                          10x1                  80  double  
%

isProperFunc = @(d)d.isProper();
inBetas = DistBeta(s.inBetaA', s.inBetaB');
inBetasProper = arrayfun(isProperFunc, inBetas);

inNormals = DistNormal(s.inNormalMeans', s.inNormalVariances');
inNormalsProper = arrayfun(isProperFunc, inNormals);

outNormals = DistNormal(s.outNormalMeans', s.outNormalVariances');
outNormalsProper = arrayfun(isProperFunc, outNormals);

% indices of proper messages
I = inBetasProper & inNormalsProper & outNormalsProper;

inBetas = DistArray(inBetas(I));
inNormals = DistArray(inNormals(I));
outNormals = DistArray(outNormals(I));

bundle = DefaultMsgBundle(outNormals, inBetas, inNormals);
serializer = BundleSerializer();
serializer.saveBundle(bundle, bundleName);

end


function genSigmoidFactorOperator()
   % convenient method to generate a sigmoid FactorOperator from results 
   % in script/saved/
   %fwFile = 'ichol_learn_ICholMapperLearner_binlogis_fw_n400_iter5_sf1_st20_ntr4000_DistBetaBuilder.mat';
   %fwFile = 'ichol_learn_ICholMapperLearner_binlogis_fw_n400_iter5_sf1_st20_ntr4000_DBetaLogBuilder.mat';
   %fwFile = 'ichol_learn_ICholMapperLearner_binlogis_fw_n400_iter5_sf1_st20_ntr6000_DBetaLogBuilder.mat';
   fwFile = 'fm_kgg_joint-irf500-orf1000-binlogis_fw_proj_n400_iter5_sf1_st20-ntr5000-DBetaLogBuilder.mat';

   %bwFile = 'ichol_learn_ICholMapperLearner_binlogis_bw_n400_iter5_sf1_st20_ntr4000_DNormalLogVarBuilder.mat';  
   %bwFile = 'ichol_learn_ICholMapperLearner_binlogis_bw_n400_iter5_sf1_st20_ntr6000_DNormalLogVarBuilder.mat';  
   bwFile = 'fm_kgg_joint-irf500-orf1000-binlogis_bw_proj_n400_iter5_sf1_st20-ntr5000-DNormalLogVarBuilder.mat';
   %foName='ichol_logbeta_n400_iter5_sf1_st20_ntr6000';
   foName='fm_kgg_joint_irf500_orf1000_proj_n400_iter5_sf1_st20_ntr5000';

   summary=sprintf(['sigmoid factor. p(x1|x2) where x1 is Beta and '...
       'x2 is Normal. fw: %s. bw: %s'], fwFile, bwFile);

   %fwPath = Expr.scriptSavedFile(fwFile);
   fwPath = Expr.expSavedFile(6, fwFile);
   %bwPath = Expr.scriptSavedFile(bwFile);
   bwPath = Expr.expSavedFile(6, bwFile);

   fo=getSigmoidFactorOperator(fwPath, bwPath, summary);
   serializer=FactorOpSerializer();
   % save operator 
   serializer.saveFactorOperator(fo, foName);

   % serialize operator for C#
   serializer.serializeFactorOperator(fo, foName);

end

function fo=getSigmoidFactorOperator(fwPath, bwPath, summary)
   % Construct a FactorOperator for sigmoid problem. 
   % This requires loading from at least two files in script/saved/ folder.
   % We need one DistMapper for forward and another for backward direction.
   %
   
   %fwPath = Expr.scriptSavedFile(fwFName) ;
   %bwPath = Expr.scriptSavedFile(bwFName);
   % expect a struct s
   % Example:
   % s = 
   %  learner_class: 'ICholMapperLearner'
   %learner_options: [1x1 Options]
   %    dist_mapper: [1x1 GenericMapper]
   %    learner_log: [1x1 struct]
   %         commit: '7608a9e0'
   %      timeStamp: [2015 1 29 13 53 44.4944]
   fwLoaded = load(fwPath);
   fwResult=fwLoaded.s;

   bwLoaded = load(bwPath);
   bwResult = bwLoaded.s;

   fw_dm=fwResult.dist_mapper;
   bw_dm=bwResult.dist_mapper;
   fo=DefaultFactorOperator({fw_dm, bw_dm}, summary);
end

function [X, Y ] = gen2DSimpleRegression(st, subsample)
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
    I = abs(st.inBetaA-1) <=1e-8 & abs(st.inBetaB-2) <= 1e-8 ...
        & ~(abs(st.inNormalMeans) <= 1e-3 & log(st.inNormalVariances)>0 );
    % The previous line is to remove messages possibly resulted from Infer.NET's 
    % truncation.
    
    n = sum(I);
    X = zeros(n, 2);
    X(:, 1) = st.inNormalMeans(I);
    X(:, 2) = log(st.inNormalVariances(I));
    Y = st.outNormalMeans(I);

    I_sub = randperm(n, min(n, subsample));
    X = X(I_sub, :);
    Y = Y(I_sub);

end


function [X, Y, Xuns, Yuns] = gen2DUncertaintyCheckData(st, subsample)
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
    % - Divide data into training and unseen set where the unseen set is from 
    % an unexplored region. Training would not help in predicting these.
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


