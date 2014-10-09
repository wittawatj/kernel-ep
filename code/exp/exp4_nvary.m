function [ ] = exp4_nvary(bunName )
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
%bunName=sprintf('sigmoid_fw_proposal_50000');
% Nicolas's data. Has almost 30000 pairs.
%bunName=sprintf('nicolas_sigmoid_bw');
%bunName=sprintf('nicolas_sigmoid_fw');
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
%bunName=sprintf('simplegauss_d1_fw_proposal_30000' );
%bunName='lds_d3_tox_20000';
se=BundleSerializer();
bundle=se.loadBundle(bunName);

n=1e4;
smallBundle=bundle.subsample(n);
%n=min(n, smallBundle.count());
inTensor=smallBundle.getInputTensorInstances();

%---------- options -----------
% number of random features for cross validation
candidate_primal_features=1000;
%candidate_primal_features=200;
% training size to vary 
trSizes = [1000,2000, 3000,4000, 5000];
%trSizes = [100 ];
teSize = 5000;
% only one fixed test set 
[bundle, teBundle] = bundle.partitionTrainTest(length(bundle)-teSize, teSize);
%teSize = 50;
% trial numbers 
%trialNums = 1:5;
trialNums = 1:10;
%trialNums = 6:10;
%trialNums = 6:20;
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
icholEGaussLearner.opt('num_ho', 3);
icholEGaussLearner.opt('chol_tol', 1e-12);
icholEGaussLearner.opt('chol_maxrank', 600);

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
%learners={ mvLearner, jointLearner, sumLearner, prodLearner };
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
    learner.opt('num_primal_features', 3000);
    learner.opt('use_multicore', true);
    learner.opt('reglist', [1e-6, 1e-4, 1e-2, 1 ]);
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
    multicore_settings.maxEvalTimeSingle = 2*60*60;
    multicoreFunc = @(ist)wrap_nvaryTestMap(ist, bundle, bunName, teBundle, relearn);
    resultCell = startmulticoremaster(multicoreFunc, stCells, multicore_settings);
    %S=[resultCell{:}];
else
    % not use multicore. Usually a bad idea as this will take much time. 
    %S={};
    for i=1:length(stCells)
        ist = stCells{i};
        s = wrap_nvaryTestMap(ist, bundle, bunName, teBundle, relearn );
        %S{i}=s;
    end
    %S=[S{:}];
end

rng(oldRng);
end

function s=wrap_nvaryTestMap(ist, bundle, bunName, teBundle, relearn)
    % wrapper to be used with startmulticoremaster(.)
    %
    % ist = input struct 
    %
    trN = ist.trN;
    teN = ist.teN;
    trialNum = ist.trialNum;
    learner = ist.learner;
    s=nvaryTestMap(trN, teN, trialNum, learner, bundle, bunName, teBundle, relearn);

end

function s=nvaryTestMap(trN, teN, trialNum, learner, bundle, bunName, teBundle, relearn)
    % teBundle = test bundle specific to a dataset is fixed 
    %  Return a struct S containing produced variables.
    
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
    % file for smaller version of the result.
    smallIden=sprintf('nvary_small-%s-%s-ntr%d-tri%d.mat', class(learner), bunName, trN, trialNum);
    fpath=Expr.expSavedFile(4, iden);
    smallFPath = Expr.expSavedFile(4, smallIden);

    if ~relearn && exist(fpath, 'file')
        % s should be loaded here. Return it
        if ~exist(smallFPath, 'file')
            try 
                load(fpath);
            catch err 
                % if load error, rm it.
                delete(fpath);
                s=[];
                return;
            end
            % save small version. Remove big objects.
            s = rmfield(s, 'teBundle');
            s = rmfield(s, 'dist_mapper');
            s = rmfield(s, 'learner_log');
            s = rmfield(s, 'out_distarray');
            s = rmfield(s, 'learner_options');
            save(smallFPath, 's');
        end
        s=[];
        return;
    end
    %/////////////////////////////

    % non-overlapping train/test sets
    [trBundle, ~] = bundle.partitionTrainTest(trN, teN);

    % learn a DistMapper
    [dm, learnerLog]=learner.learnDistMapper(trBundle);

    % test on the test MsgBundle
    %keyboard
    outDa = dm.mapMsgBundle(teBundle);
    assert(isa(outDa, 'DistArray'));

    % save everything
    commit=GitTool.getCurrentCommit();
    timeStamp=clock();

    trueOutDa = teBundle.getOutBundle();
    divTester = DivDistMapperTester(dm);
    divTester.opt('div_function', 'KL');
    divs = divTester.getDivergence(outDa, trueOutDa);

    % Return a struct 
    s=struct();
    s.divs = divs;
    s.improper_count = sum(isnan(divs));
    s.learner_class=class(learner);
    % type Options
    s.learner_options=learner.options;
    s.result_path=fpath;
    s.dist_mapper=dm;
    s.learner_log=learnerLog;
    s.out_distarray=outDa;
    s.commit=commit;
    s.timeStamp=timeStamp;

    s.trN = trN;
    s.teN = teN;
    s.trialNum=trialNum;
    s.teBundle = teBundle;
    s.bundleName=bunName;

    save(fpath, 's');
    % save small version. Remove big objects.
    s = rmfield(s, 'teBundle');
    s = rmfield(s, 'dist_mapper');
    s = rmfield(s, 'learner_log');
    s = rmfield(s, 'out_distarray');
    s = rmfield(s, 'learner_options');
    save(smallFPath, 's');
end

