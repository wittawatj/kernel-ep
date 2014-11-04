function [ ] = ichol_learn_cmaes( )
% test learning operator with incomplete Cholesky. Optimize parameters with CMA-ES
%

seed=32;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
%bunName='sigmoid_bw_proposal_5000';
%bunName = 'sigmoid_fw_proposal_5000';
% Nicolas's data. Has almost 30000 pairs.
bunName=sprintf('nicolas_sigmoid_bw');
%bunName=sprintf('nicolas_sigmoid_fw');
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
%bunName=sprintf('simplegauss_d1_fw_proposal_30000' );
%bunName='lds_d3_tox_3000';
bundle=se.loadBundle(bunName);

%n=5000;
%n=25000;
%[trBundle, teBundle] = bundle.partitionTrainTest(40000, 10000);
[trBundle, teBundle] = bundle.partitionTrainTest(6000, 4000);
%[trBundle, teBundle] = bundle.partitionTrainTest(2000, 3000);

%---------- options -----------
learner=ICholMapperLearner();
%inTensor = bundle.getInputTensorInstances();

od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();

% set my options
learner.opt('seed', seed);
learner.opt('out_msg_distbuilder', DNormalLogVarBuilder());
%learner.opt('out_msg_distbuilder', DNormalSDBuilder());
learner.opt('use_cmaes', true);
learner.opt('kernel_mode', 'kggauss');
learner.opt('num_ho', 3);
learner.opt('ho_train_size', 2000);
learner.opt('ho_test_size', 2000);
learner.opt('chol_tol', 1e-12);
learner.opt('chol_maxrank_train', 100);
learner.opt('chol_maxrank', 1000);
learner.opt('use_multicore', true);
learner.opt('separate_outputs', true)

s=learnMap(learner, trBundle, teBundle, bunName);
n=length(trBundle)+length(teBundle);
iden=sprintf('ichol_learn_cmaes_%s_%s_%d.mat', class(learner), bunName, n);
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

