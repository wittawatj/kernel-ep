function [ ] = exp5_nfvary(bunName )
% Vary the number of random features and train an operator for multiple trials.
% One saved file for one (nf, trial, method, data). 
% Combine them later.
%  * For each seed and dataset, one fixed test set 
%  * #random features during CV is fixed. 
%  * no incomplete Cholesky because it does not use random features.
%

seed=5;
oldRng=rng();
rng(seed);

% true to relearn everything. This is for overriding the results obtained 
% in the past. 
relearn=false;
%relearn=true;

%bunName=sprintf('sigmoid_bw_proposal_50000');
%bunName=sprintf('sigmoid_bw_proposal_2000' );
%bunName=sprintf('sigmoid_fw_proposal_50000');
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
candidate_primal_features=1000;
%candidate_primal_features=200;

% fixed training size 
trSize = 5000;
%trSize = 1e3;
teSize = 5000;
% only one test bundle 
[bundle, teBundle] = bundle.partitionTrainTest(length(bundle)-teSize , teSize);
%teSize = 50;
% trial numbers 
%trialNums = 1:5;
trialNums = 1:10;

% number of random features to vary 
%nfs = 1000:2000:1e4;
nfs = [20, 50, 100, 200, 100, 500, 1000, 1500, 2000];
%nfs = [100, 200];

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

% learner-specific options
mvCandidates=RandFourierGaussMVMap.candidatesFlatMedf(inTensor, medf, ...
    candidate_primal_features, 1500);
jointCandidates=RFGJointEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 5000);
sumCandidates=RFGSumEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 5000);
prodCandidates=RFGProductEProdMap.candidatesAvgCov(inTensor, medf, ...
    candidate_primal_features, 5000);
% set candidates for each learner
mvLearner.opt('featuremap_candidates', mvCandidates);
jointLearner.opt('featuremap_candidates', jointCandidates);
sumLearner.opt('featuremap_candidates', sumCandidates);
prodLearner.opt('featuremap_candidates', prodCandidates);

learners={ mvLearner, jointLearner, sumLearner, prodLearner };
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
    learner.opt('use_multicore', true);
    learner.opt('reglist', [1e-6, 1e-4, 1e-2, 1 ]);
end

% Generate combinations to try 
totalCombs = length(nfs)*length(trialNums)*length(learners);
stCells = cell(1, totalCombs);
for i=1:length(nfs)
    for j=1:length(trialNums)
        for l=1:length(learners)
            ci = sub2ind([length(nfs), length(trialNums), length(learners)], i, j, l);
            st = struct();
            st.nf = nfs(i);
            st.trN = trSize;
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
    multicoreFunc = @(ist)wrap_nfvaryTestMap(ist, bundle, bunName, teBundle, relearn);
    resultCell = startmulticoremaster(multicoreFunc, stCells, multicore_settings);
    S=[resultCell{:}];
else
    % not use multicore. Usually a bad idea as this will take much time. 
    S={};
    for i=1:length(stCells)
        ist = stCells{i};
        s = wrap_nfvaryTestMap(ist, bundle, bunName, teBundle, relearn );
        S{i}=s;
    end
    S=[S{:}];
end

rng(oldRng);
end

function s=wrap_nfvaryTestMap(ist, bundle, bunName, teBundle, relearn)
    % wrapper to be used with startmulticoremaster(.)
    %
    % ist = input struct 
    %
    trN = ist.trN;
    teN = ist.teN;
    trialNum = ist.trialNum;
    learner = ist.learner;
    nf = ist.nf;
    s=nfvaryTestMap(trN, teN, nf, trialNum, learner, bundle, bunName, teBundle, relearn);

end

function s=nfvaryTestMap(trN, teN, nf, trialNum, learner, bundle, bunName, teBundle, relearn)
    % teBundle = test bundle specific to a dataset is fixed 
    %  * nf = number of random features to vary 
    % Return a struct S containing produced variables.
    
    rng(trialNum);

    learner.opt('seed', trialNum);

    assert(isa(learner, 'DistMapperLearner'));
    assert(trN > 0);
    assert(teN > 0);
    assert(trialNum > 0);
    assert(nf > 0);

    iden=sprintf('nfvary-%s-%s-nf%d-tri%d.mat', class(learner), bunName, nf, trialNum);
    % file for smaller version of the result.
    smallIden=sprintf('nfvary_small-%s-%s-nf%d-tri%d.mat', class(learner), bunName, nf, trialNum);
    fpath=Expr.expSavedFile(5, iden);
    smallFPath = Expr.expSavedFile(5, smallIden);

    if ~relearn && exist(fpath, 'file')
        % s should be loaded here. Return it
        if ~exist(smallFPath, 'file')
            try 
                load(fpath)
            catch err 
                % if load error, rm it 
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
    % vary the number of features
    learner.opt('num_primal_features', nf);

    % non-overlapping train/test sets
    [trBundle, ~] = bundle.partitionTrainTest(trN, teN);

    % learn a DistMapper
    [dm, learnerLog]=learner.learnDistMapper(trBundle);
    
    % test on the test MsgBundle
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
    % number of features
    s.nf = nf;
    s.trialNum=trialNum;
    s.teBundle=teBundle;
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

