%function [ ] = predict_balaji_msg_set( )
%PREDICT_BALAJI_MSG_SET This scripts accompanies gen_balaji_msg_set. It predicts 
%the outgoing messages of the incoming messages in the generated dataset using 
%kernel-EP's GP predictor.
%

resultFname = 'fm_kgg_joint-irf500-orf1000-binlogis_bw_proj_n400_iter5_sf1_st20-ntr5000-DNormalLogVarBuilder.mat';
resultPath = Expr.expSavedFile(6, resultFname);
% load the dataset 
dataName = 'balaji_msg_set-par500000.mat';
dataPath = fullfile(Global.getScriptFolder(), 'logistic_msg', dataName);
D = load(dataPath);
%D = 
%          Ytr: [4012x2 double]
%         Yte1: [500x2 double]
%         Yte2: [500x2 double]
%          Xtr: [4012x4 double]
%         Xte1: [500x4 double]
%         Xte2: [500x4 double]
%    particles: 500000
%    timeStamp: [2015 6 2 20 29 23.3784]
% convert the parameters to Distribution objects
Btr = DistBeta(D.Xtr(:, 1)', D.Xtr(:, 2)');
Gtr = DistNormal(D.Xtr(:, 3)', exp(-D.Xtr(:, 4))' );

Bte1 = DistBeta(D.Xte1(:, 1)', D.Xte1(:, 2)');
Gte1 = DistNormal(D.Xte1(:, 3)', exp(-D.Xte1(:, 4))' );

Bte2 = DistBeta(D.Xte2(:, 1)', D.Xte2(:, 2)');
Gte2 = DistNormal(D.Xte2(:, 3)', exp(-D.Xte2(:, 4))' );

ex= load(resultPath);
dm = ex.s.dist_mapper;
%ex = 
%               trBundle: [1x1 DefaultMsgBundle]
%               teBundle: [1x1 DefaultMsgBundle]
%    out_msg_distbuilder: [1x1 DNormalLogVarBuilder]
%                      s: [1x1 struct]
%              timeStamp: [2015 4 21 13 24 10.5413] 
% The distribution mapper (predictor).
%
%>> ex.s
%ans = 
%      learner_class: 'RFGJointKGGLearner'
%    learner_options: [1x1 Options]
%        dist_mapper: [1x1 UAwareGenericMapper]
%        learner_log: [1x1 struct]
%             commit: '275f5908'
%          timeStamp: [2015 4 21 13 24 10.5385]
[Otr, Utr] = dm.mapDistsAndU(Btr, Gtr);
[Ote1, Ute1] = dm.mapDistsAndU(Bte1, Gte1);
[Ote2, Ute2] = dm.mapDistsAndU(Bte2, Gte2);

%% 
OMtr = [Otr.mean]';
OLPtr = [-log([Otr.variance])]';
LogUtr = log(Utr)';

OMte1 = [Ote1.mean]';
OLPte1 = [-log([Ote1.variance])]';
LogUte1 = log(Ute1)';

OMte2 = [Ote2.mean]';
OLPte2 = [-log([Ote2.variance])]';
LogUte2 = log(Ute2)';

saveName = sprintf('predict_%s', dataName);
savePath = fullfile(Global.getScriptFolder(), 'logistic_msg', saveName);
save(savePath, 'OMtr', 'OLPtr', 'LogUtr', 'OMte1', 'OLPte1', 'LogUte1', ...
    'OMte2', 'OLPte2', 'LogUte2');
%end

