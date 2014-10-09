function [ ] = exp3_eprod_heuristic( )
% This experiment is mainly for testing a heuristic in choosing kernel parameter
% for expected product kernel. The heuristic is to use the average of all 
% covariance matrices for all DistNormal messages. Take the diagonal as kernel 
% parameter.
%

seed=121+1;
oldRng=rng();
rng(seed);
% true to relearn everything 
relearn=false;
%relearn=true;

se=BundleSerializer();
sample_cond_msg=false;
if sample_cond_msg
    anno='samcond';
else
    anno='proposal';
end
bunName=sprintf('sigmoid_bw_%s_25000', anno);
%bunName=sprintf('sigmoid_bw_%s_2000', anno);
% Nicolas's data. Has almost 30000 pairs.
%bunName=sprintf('nicolas_sigmoid_bw');
%bunName=sprintf('nicolas_sigmoid_fw');
%bunName=sprintf('simplegauss_d1_bw_samcond_30000' );
%bunName=sprintf('simplegauss_d1_fw_samcond_30000' );
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
%bunName=sprintf('simplegauss_d1_fw_proposal_30000' );
%bunName='lds_d3_tox_3000';
bundle=se.loadBundle(bunName);

n=6250;
%n=25000;
smallBundle=bundle.subsample(n);
n=min(n, smallBundle.count());
% train 80%
[trBundle, teBundle]=smallBundle.splitTrainTest(.8);
trTensor=trBundle.getInputTensorInstances();

%---------- options -----------
candidate_primal_features=2000;
%candidate_primal_features=200;
% median factors
medf=[1/40, 1/20, 1/10, 1/5, 1/3, 1/2, 1, 2, 3, 5, 10, 20, 40];
% run multicore
use_multicore=true;
%use_multicore=false;
%----------
% learners
mvLearner=RFGMVMapperLearner();
jointLearner=RFGJointEProdLearner();
sumLearner=RFGSumEProdLearner();
prodLearner=RFGProductEProdLearner();
icholEGaussLearner=ICholMapperLearner();
icholEGaussLearner.opt('num_ho', 3);
ntr=length(trBundle);
icholEGaussLearner.opt('ho_train_size', min(2e4, floor(0.7*ntr)) );
% internally form tr x te matrix. ho_test_size cannot be too large.
icholEGaussLearner.opt('ho_test_size', min(3e3, floor(0.3*ntr)) );
icholEGaussLearner.opt('chol_tol', 1e-12);
icholEGaussLearner.opt('chol_maxrank', 600);

% learner-specific options
mvCandidates=RandFourierGaussMVMap.candidatesFlatMedf(trTensor, medf, ...
    candidate_primal_features, 1500);
jointCandidates=RFGJointEProdMap.candidatesAvgCov(trTensor, medf, ...
    candidate_primal_features, 5000);
sumCandidates=RFGSumEProdMap.candidatesAvgCov(trTensor, medf, ...
    candidate_primal_features, 5000);
prodCandidates=RFGProductEProdMap.candidatesAvgCov(trTensor, medf, ...
    candidate_primal_features, 5000);
icholEGaussCandidates=KEGaussian.productCandidatesAvgCov(trTensor, medf, 5000);
% set candidates for each learner
mvLearner.opt('featuremap_candidates', mvCandidates);
jointLearner.opt('featuremap_candidates', jointCandidates);
sumLearner.opt('featuremap_candidates', sumCandidates);
prodLearner.opt('featuremap_candidates', prodCandidates);
icholEGaussLearner.opt('kernel_candidates', icholEGaussCandidates);

%learners={ mvLearner, jointLearner, sumLearner, prodLearner, icholEGaussLearner};
learners={mvLearner, jointLearner};
%learners={ prodLearner};
%learners={icholEGaussLearner};

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

    % set my options
    learner.opt('seed', 221+1);
    %learner.opt('num_primal_features', 1000);
    learner.opt('use_multicore', use_multicore);
    learner.opt('num_primal_features', 3000);
    %learner.opt('use_multicore', true);
    learner.opt('reglist', [1e-6, 1e-4, 1e-2, 1]);
end

if use_multicore
    gop=globalOptions();
    multicore_settings.multicoreDir= gop.multicoreDir;                    
    multicore_settings.maxEvalTimeSingle = 2*60*60;
    learnMapfunc=@(l)learnMap(l, trBundle, teBundle, bunName, relearn);
    resultCell = startmulticoremaster(learnMapfunc, learners, multicore_settings);
    S=[resultCell{:}];
else
    % not use multicore 
    S={};
    for i=1:length(learners)
        l=learners{i};
        s=learnMap(l, trBundle, teBundle, bunName, relearn);
        S{i}=s;
    end
    S=[S{:}];
end

n=length(trBundle)+length(teBundle);
iden=sprintf('%d_learners_%s_%d.mat', length(learners), bunName, n);
fpath=Expr.expSavedFile(3, iden);

commit=GitTool.getCurrentCommit();
timeStamp=clock();
save(fpath, 'S', 'learners', 'commit', 'timeStamp', 'trBundle', 'teBundle');

rng(oldRng);
end

function s=learnMap(learner, trBundle, teBundle, bunName, relearn)
    % run the specified learner. 
    % Return a struct S containing produced variables.

    assert(isa(learner, 'DistMapperLearner'));
    assert(isa(trBundle, 'MsgBundle'));

    n=length(trBundle)+length(teBundle);
    iden=sprintf('%s_%s_%d.mat', class(learner), bunName, n);
    fpath=Expr.expSavedFile(3, iden);

    if ~relearn && exist(fpath, 'file')
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
    %keyboard
    [Divs, outDa]=divTester.testDistMapper(teBundle);
    assert(isa(outDa, 'DistArray'));

    % Check improper messages
    impTester=ImproperDistMapperTester(dm);
    impOut=impTester.testDistMapper(teBundle);

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
    s.div_tester=divTester;
    s.divs=Divs;
    s.out_distarray=outDa;
    s.imp_tester=impTester;
    s.imp_out=impOut;
    s.commit=commit;
    s.timeStamp=timeStamp;

    save(fpath, 's');
end

