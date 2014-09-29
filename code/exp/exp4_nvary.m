function [ ] = exp4_nvary( )
% Vary the training size n and train an operator for multiple trials.
% One saved file for one (n, trial, method, data). 
% Combine them later.
%  * For each seed and dataset, one fixed test set 
%

seed=2;
oldRng=rng();
rng(seed);
% true to relearn everything. This is for overriding the results obtained 
% in the past. 
relearn=false;
%relearn=true;

sample_cond_msg=false;
if sample_cond_msg
    anno='samcond';
else
    anno='proposal';
end
%bunName=sprintf('sigmoid_bw_%s_50000', anno);
%bunName=sprintf('sigmoid_bw_%s_2000', anno);
bunName=sprintf('sigmoid_fw_proposal_50000');
% Nicolas's data. Has almost 30000 pairs.
%bunName=sprintf('nicolas_sigmoid_bw');
%bunName=sprintf('nicolas_sigmoid_fw');
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
%bunName=sprintf('simplegauss_d1_fw_proposal_30000' );
se=BundleSerializer();
bundle=se.loadBundle(bunName);

n=1e4;
smallBundle=bundle.subsample(n);
%n=min(n, smallBundle.count());
inTensor=smallBundle.getInputTensorInstances();

%---------- options -----------
% number of random features for cross validation
candidate_primal_features=2000;
%candidate_primal_features=200;
% training size to vary 
trSizes = 2000:2000:1e4;
%trSizes = [100 ];
teSize = 5000;
%teSize = 50;
% trial numbers 
%trialNums = 1:5;
trialNums = 6:20;
% fixed test size for each training size 

% median factors
medf=[1/30, 1/10, 1/5,  1/2, 1, 2,  5, 10, 30];
%medf=[1/2, 1, 2];
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
icholEGaussLearner.opt('num_ho', 5);
icholEGaussLearner.opt('chol_tol', 1e-12);
icholEGaussLearner.opt('chol_maxrank', 800);

% learner-specific options
mvCandidates=RandFourierGaussMVMap.candidatesFlatMedf(inTensor, medf, ...
    candidate_primal_features, 1500);
jointCandidates=RFGJointEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 5000);
sumCandidates=RFGSumEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 5000);
prodCandidates=RFGProductEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 5000);
icholEGaussCandidates=KEGaussian.productCandidatesAvgCov(inTensor, medf, 5000);
% set candidates for each learner
mvLearner.opt('featuremap_candidates', mvCandidates);
jointLearner.opt('featuremap_candidates', jointCandidates);
sumLearner.opt('featuremap_candidates', sumCandidates);
prodLearner.opt('featuremap_candidates', prodCandidates);
icholEGaussLearner.opt('kernel_candidates', icholEGaussCandidates);

learners={ mvLearner, jointLearner, sumLearner, prodLearner, icholEGaussLearner};
%learners={ prodLearner};
%learners={icholEGaussLearner};

for i=1:length(learners)
    learner=learners{i};
    od=learner.getOptionsDescription();
    %display(' Learner options: ');
    %od.show();
    % only used by RFGMVMapperLearner
    % The following options are not needed as candidates are already specified.
    %learner.opt('mean_med_factors', [1/4, 1, 1/4]);
    %learner.opt('variance_med_factors', [1/4, 1, 1/4]);
    %learner.opt('med_factors', [1]);
    learner.opt('candidate_primal_features', candidate_primal_features);

    % set my options
    %learner.opt('num_primal_features', 1000);
    %learner.opt('use_multicore', use_multicore);
    learner.opt('num_primal_features', 10000);
    learner.opt('use_multicore', true);
    learner.opt('reglist', [1e-4, 1e-2, 1, 100]);
end

% Generate combinations to try 
totalCombs = length(trSizes)*length(trialNums)*length(learners);
stCells = cell(1, totalCombs);
for i=1:length(trSizes)
    for j=1:length(trialNums)
        for l=1:length(learners)
            ci = sub2ind([length(trSizes), length(trialNums), length(learners)], i, j, l);
            st = struct();
            st.trN = trSizes(i);
            st.teN = teSize;
            st.trialNum = trialNums(j);
            st.learner = learners{l};

            stCells{ci} = st;
        end
    end

end

if use_multicore
    gop=globalOptions();
    multicore_settings.multicoreDir= gop.multicoreDir;                    
    multicoreFunc = @(ist)wrap_nvaryTestMap(ist, bundle, bunName, relearn);
    resultCell = startmulticoremaster(multicoreFunc, stCells, multicore_settings);
    S=[resultCell{:}];
else
    % not use multicore. Usually a bad idea as this will take much time. 
    S={};
    for i=1:length(stCells)
        ist = stCells{i};
        s = wrap_nvaryTestMap(ist, bundle, bunName, relearn );
        S{i}=s;
    end
    S=[S{:}];
end

rng(oldRng);
end

function s=wrap_nvaryTestMap(ist, bundle, bunName, relearn)
    % wrapper to be used with startmulticoremaster(.)
    %
    % ist = input struct 
    %
    trN = ist.trN;
    teN = ist.teN;
    trialNum = ist.trialNum;
    learner = ist.learner;
    s=nvaryTestMap(trN, teN, trialNum, learner, bundle, bunName, relearn);

end

function s=nvaryTestMap(trN, teN, trialNum, learner, bundle, bunName, relearn)
    % teBundle = test bundle specific to a dataset is fixed for all n 
    % run the specified learner. 
    % Return a struct S containing produced variables.
    
    if isa(learner, 'ICholMapperLearner')
        learner.opt('ho_train_size', min(2e4, floor(0.7*trN)) );
        % internally form tr x te matrix. ho_test_size cannot be too large.
        learner.opt('ho_test_size', min(3e3, floor(0.3*trN)) );
    end
    rng(trialNum);

    learner.opt('seed', trialNum);

    assert(isa(learner, 'DistMapperLearner'));
    assert(trN > 0);
    assert(teN > 0);
    assert(trialNum > 0);

    iden=sprintf('nvary-%s-%s-ntr%d-tri%d.mat', class(learner), bunName, trN, trialNum);
    fpath=Expr.expSavedFile(4, iden);

    if ~relearn && exist(fpath, 'file')
        load(fpath);
        % s should be loaded here. Return it
        return;
    end
    %/////////////////////////////

    % non-overlapping train/test sets
    [trBundle, teBundle] = bundle.partitionTrainTest(trN, teN);

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

    s.trN = trN;
    s.teN = teN;
    s.trialNum=trialNum;
    s.bundleName=bunName;

    save(fpath, 's');
end

