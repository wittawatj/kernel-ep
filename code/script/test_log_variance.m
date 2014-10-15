function [ ] = test_log_variance( )
% Test operator when output is log variance

seed=23;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
% output has to be 1d Gaussians
bunName='sigmoid_bw_proposal_5000';
% Nicolas's data. Has almost 30000 pairs.
%bunName=sprintf('nicolas_sigmoid_bw');
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
bundle=se.loadBundle(bunName);

%n=5000;
[trBundle, teBundle] = bundle.partitionTrainTest(2000, 2000);
%[trBundle, teBundle] = bundle.partitionTrainTest(200, 200);

%---------- options -----------
candidate_primal_features=500;

%----------
learner=RFGJointEProdLearner();
%learner=RFGProductEProdLearner();
%learner=RFGMVMapperLearner();

od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();
learner.opt('out_msg_distbuilder', DNormalLogVarBuilder() )
learner.opt('candidate_primal_features', candidate_primal_features);
% ///// use CMA-ES //////
learner.opt('use_cmaes', true);

% set my options
learner.opt('seed', 40);
%learner.opt('num_primal_features', 1000);
learner.opt('use_multicore', false);
learner.opt('num_primal_features', 1000);

s=learnMap(learner, trBundle, teBundle, bunName);
n=length(trBundle)+length(teBundle);
iden=sprintf('logvariance_%s_%s_%d.mat', class(learner), bunName, n);
fpath=Expr.scriptSavedFile(iden);

timeStamp=clock();
save(fpath, 's', 'timeStamp', 'trBundle', 'teBundle');

rng(oldRng);

end

function s=learnMap(learner, trBundle, teBundle, bunName)
    % run the specified learner. 
    % Return a struct S containing produced variables.

    assert(isa(learner, 'DistMapperLearner'));
    assert(isa(trBundle, 'MsgBundle'));

    n=length(trBundle)+length(teBundle);

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
    s.dist_mapper=dm;
    s.learner_log=learnerLog;
    s.div_tester=divTester;
    s.divs=Divs;
    s.out_distarray=outDa;
    s.imp_tester=impTester;
    s.imp_out=impOut;
    s.commit=commit;
    s.timeStamp=timeStamp;

end

