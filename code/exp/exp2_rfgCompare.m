function [ ] = exp2_rfgCompare( )
% Compare many Random Fourier feature maps
%

seed=121+1;
oldRng=rng();
rng(seed);
% true to relearn everything 
relearn=false;

se=BundleSerializer();
sample_cond_msg=false;
if sample_cond_msg
    anno='samcond';
else
    anno='proposal';
end
%bunName=sprintf('sigmoid_bw_%s_25000', anno);
%bunName=sprintf('sigmoid_bw_%s_2000', anno);
% Nicolas's data. Has almost 30000 pairs.
bunName=sprintf('nicolas_sigmoid_bw');
bundle=se.loadBundle(bunName);

%n=200;
n=25000;
smallBundle=bundle.subsample(n);
n=min(n, smallBundle.count());
% train 80%
[trBundle, teBundle]=smallBundle.splitTrainTest(.8);

learners={RFGMVMapperLearner(), ...
    RFGJointEProdLearner(),...
    RFGSumEProdLearner()};
%learners={ RFGSumEProdLearner()};

% run multicore
use_multicore=true;
%use_multicore=false;
if use_multicore
    gop=globalOptions();
    multicore_settings.multicoreDir= gop.multicoreDir;                    
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
fpath=Expr.expSavedFile(2, iden);

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
    fpath=Expr.expSavedFile(2, iden);
    od=learner.getOptionsDescription();

    if ~relearn && exist(fpath, 'file')
        load(fpath);
        % s should be loaded here. Return it
        return;
    end

    display(' Learner options: ');
    od.show();
    % only used by RFGMVMapperLearner
    learner.opt('mean_med_factors', [1/4, 1, 1/4]);
    learner.opt('variance_med_factors', [1/4, 1, 1/4]);

    % set my options
    learner.opt('seed', 221+1);
    %learner.opt('med_factors', [1]);
    %learner.opt('num_primal_features', 2000);
    %learner.opt('candidate_primal_features', 200);
    %learner.opt('use_multicore', false);
    learner.opt('med_factors', 10.^(-2:2));
    learner.opt('num_primal_features', 10000);
    learner.opt('candidate_primal_features', 2000);
    learner.opt('use_multicore', true);
    learner.opt('reglist', [1e-4, 1e-2, 1]);

    % learn a DistMapper
    [dm, learnerLog]=learner.learnDistMapper(trBundle);

    % test the learned DistMapper dm
    % KL or Hellinger
    divTester=DivDistMapperTester(dm);
    divTester.opt('div_function', 'KL'); 
    % test on the test MsgBundle
    %keyboard
    [Helling, outDa]=divTester.testDistMapper(teBundle);
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
    s.helling=Helling;
    s.out_distarray=outDa;
    s.imp_tester=impTester;
    s.imp_out=impOut;
    s.commit=commit;
    s.timeStamp=timeStamp;

    save(fpath, 's');
end
