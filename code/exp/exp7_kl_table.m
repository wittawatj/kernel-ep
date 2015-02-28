function [ ] = exp7_kl_table( )
%EXP7_KL_TABLE This experiment is used to generate a table comparing different learners.

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
%bunName = 'binlogis_fw_n400_iter5_sf1_st20';
%bunName = 'binlogis_bw_n400_iter5_sf1_st20';

bunName = 'binlogis_bw_proj_n400_iter5_sf1_st20';
%bunName = 'binlogis_bw_n400_iter20_s1';
%bunName = 'binlogis_fw_proj_n400_iter5_sf1_st20';
%bunName = 'binlogis_bw_n400_iter20_s1';
%bunName = 'binlogis_fw_n1000_iter5_s1';
bundle=se.loadBundle(bunName);

%n=5000;
%n=25000;
[trBundle, teBundle] = bundle.partitionTrainTest(6000, 2000);
%[trBundle, teBundle] = bundle.partitionTrainTest(500, 500);
%[trBundle, teBundle] = bundle.partitionTrainTest(3000, 1000);

%---------- options -----------
inTensor = trBundle.getInputTensorInstances();
% median factors 
medf = [1/10, 1/5, 1/3, 1/2, 1/5, 1, 2, 5, 10];
%medf = [1/2, 1/5, 1];
%num_primal_features = 300; 
%num_inner_primal_features = 200;
%candidate_primal_features = 300;
%candidate_inner_primal_features = 200;
%
num_primal_features = 1000; 
num_inner_primal_features = 500;
candidate_primal_features = 500;
candidate_inner_primal_features = 300;
%out_msg_distbuilder = DNormalSDBuilder();
%out_msg_distbuilder = DistNormalBuilder();
%out_msg_distbuilder = DistBetaBuilder();

out_msg_distbuilder = DNormalLogVarBuilder();
%out_msg_distbuilder = DBetaLogBuilder();

% learners
mvLearner=RFGMVMapperLearner();
jointLearner=RFGJointEProdLearner();
sumLearner=RFGSumEProdLearner();
prodLearner=RFGProductEProdLearner();
jointKggLearner = RFGJointKGGLearner();
icholProdKegLearner =ICholMapperLearner();
icholProdKggLearner =ICholMapperLearner();
icholSumKggLearner =ICholMapperLearner();
icholJointKggLearner =ICholMapperLearner();

use_multicore = true;
% learner-specific options
mvCandidates=RandFourierGaussMVMap.candidatesFlatMedf(inTensor, medf, ...
    candidate_primal_features, 1500);
jointCandidates=RFGJointEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 2000);
sumCandidates=RFGSumEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 2000);
prodCandidates=RFGProductEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 2000);
jointKggCandidates = RFGJointKGG.candidatesAvgCov(inTensor, medf, ...
    candidate_inner_primal_features, candidate_primal_features, 2000);
icholProdKegCandidates =KEGaussian.productCandidatesAvgCov(inTensor, medf, 5000);
icholProdKggCandidates = KGGaussian.productCandidatesAvgCov(inTensor, medf, 2000);
icholSumKggCandidates = KGGaussian.ksumCandidatesAvgCov(inTensor, medf, 2000);
icholJointKggCandidates = KGGaussianJoint.candidatesAvgCov(inTensor, medf, 2000);
%
% set candidates for each learner
mvLearner.opt('featuremap_candidates', mvCandidates);
jointLearner.opt('featuremap_candidates', jointCandidates);
sumLearner.opt('featuremap_candidates', sumCandidates);
prodLearner.opt('featuremap_candidates', prodCandidates);
jointKggLearner.opt('featuremap_candidates', jointKggCandidates);
icholProdKegLearner.opt('kernel_candidates', icholProdKegCandidates );
icholProdKggLearner.opt('kernel_candidates', icholProdKggCandidates);
icholSumKggLearner.opt('kernel_candidates', icholSumKggCandidates);
icholJointKggLearner.opt('kernel_candidates', icholJointKggCandidates);

learners={ mvLearner, jointLearner, sumLearner, prodLearner, ...
    icholProdKegLearner, icholProdKggLearner, icholSumKggLearner, icholJointKggLearner};
%learners={mvLearner, jointLearner};
%learners={ prodLearner};
%learners={icholEGaussLearner};

%
% ----- options for RFGJointKGGLearner ----
jointKggLearner.opt('num_primal_features', num_primal_features);
jointKggLearner.opt('num_inner_primal_features', num_inner_primal_features);
jointKggLearner.opt('featuremap_candidates', jointKggCandidates);
%
icholLearners = { icholProdKegLearner, icholProdKggLearner, ...
    icholSumKggLearner, icholJointKggLearner};
for i=1:length(icholLearners)
    ilearner = icholLearners{i};
    % --- options for ICholMapperLearner ----
    ilearner.opt('num_ho', 3);
    ilearner.opt('ho_train_size', 1000);
    ilearner.opt('ho_test_size', 1500);
    %ilearner.opt('ho_train_size', 100);
    %ilearner.opt('ho_test_size', 100);
    ilearner.opt('chol_tol', 1e-8);
    ilearner.opt('chol_maxrank_train', 100);
    ilearner.opt('chol_maxrank', 800 );
    ilearner.opt('separate_outputs', true);
end

% ----- options for all learners -------------
for i=1:length(learners)
    learner=learners{i};
    od=learner.getOptionsDescription();
    display(' Learner options: ');
    od.show();
    % only used by RFGMVMapperLearner
    % The following options are not needed as candidates are already specified.
    %learner.opt('mean_med_factors', [1/4, 1, 1/4]);
    %learner.opt('variance_med_factors', [1/4, 1, 1/4]);
    %learner.opt('med_factors', [1]);
    learner.opt('candidate_primal_features', candidate_primal_features);

    learner.opt('seed', seed);
    learner.opt('out_msg_distbuilder', out_msg_distbuilder);
    %learner.opt('num_primal_features', 1000);
    learner.opt('use_multicore', use_multicore);
    learner.opt('num_primal_features', num_primal_features);
    %learner.opt('use_multicore', true);
    learner.opt('reglist', [1e-4, 1e-3, 1e-2, 1e-1, 1]);
end

if use_multicore
    gop=globalOptions();
    multicore_settings.multicoreDir= gop.multicoreDir;                    
    multicore_settings.maxEvalTimeSingle = 2*60*60;
    learnMapfunc=@(l)learnMap(l, trBundle, teBundle, bunName );
    resultCell = startmulticoremaster(learnMapfunc, learners, multicore_settings);
    S=[resultCell{:}];
else
    % not use multicore 
    S={};
    for i=1:length(learners)
        l=learners{i};
        s=learnMap(l, trBundle, teBundle, bunName);
        S{i}=s;
    end
    S=[S{:}];
end

n=length(trBundle)+length(teBundle);
iden=sprintf('%d_learners_%s_%d.mat', length(learners), bunName, n);
fpath=Expr.expSavedFile(7, iden);

timeStamp=clock();
save(fpath, 'S', 'learners', 'timeStamp', 'trBundle', 'teBundle');

rng(oldRng);


end

function s=learnMap(learner, trBundle, teBundle, bunName)
    % run the specified learner. 
    % Return a struct S containing produced variables.

    assert(isa(learner, 'DistMapperLearner'));
    assert(isa(trBundle, 'MsgBundle'));

    ntr=length(trBundle);
    iden=sprintf('%s_%s_ntr%d.mat', class(learner), bunName, ntr);
    fpath=Expr.expSavedFile(7, iden);
    if exist(fpath, 'file')
        load(fpath);
        % s should be loaded here. Return it
        return;
    end

    % learn a DistMapper
    [dm, learnerLog]=learner.learnDistMapper(trBundle);

    % test the learned DistMapper dm
    % KL or Hellinger
    divTester=DivDistMapperTester(dm);
    divTester.opt('div_function', 'KL'); 
    % test on the test MsgBundle
    outDa=dm.mapMsgBundle(teBundle);
    assert(isa(outDa, 'DistArray'));
    trueOutDa=teBundle.getOutBundle();
    Divs=divTester.getDivergence(outDa, trueOutDa);

    % save everything
    commit=GitTool.getCurrentCommit();
    timeStamp=clock();

    % Return a struct 
    s=struct();
    s.learner_class=class(learner);
    % type Options
    s.learner_options=learner.options;
    s.result_path=fpath;
    s.dist_mapper=dm;
    s.learner_log=learnerLog;
    s.divs=Divs;
    s.out_distarray=outDa;
    s.commit=commit;
    s.timeStamp=timeStamp;

    save(fpath, 's');
end

