function [ ] = test_kllosslearner( )
% test KLLossLearner
%

seed=32;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
bunName='sigmoid_bw_proposal_10000';
%bunName = 'sigmoid_fw_proposal_5000';
% Nicolas's data. Has almost 30000 pairs.
%bunName=sprintf('nicolas_sigmoid_bw');
%bunName=sprintf('nicolas_sigmoid_fw');
%bunName=sprintf('simplegauss_d1_bw_samcond_30000' );
%bunName=sprintf('simplegauss_d1_fw_samcond_30000' );
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
%bunName=sprintf('simplegauss_d1_fw_proposal_30000' );
%bunName='lds_d3_tox_3000';
bundle=se.loadBundle(bunName);

%n=5000;
%n=25000;
%[trBundle, teBundle] = bundle.partitionTrainTest(3000, 2000);
[trBundle, teBundle] = bundle.partitionTrainTest(2000, 5000);
%[trBundle, teBundle] = bundle.partitionTrainTest(500, 300);


%----------
learner=KLLossLearner();

od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();

% set my options
% The number of random features during the optimization has to be the same 
% as the test-time number because we directly optimize W which depends on 
% nf.
nf = 20000;
% parameters here are obtained by what are chosen by kernel_param_cmaes
%feature_map = RFGJointEProdMap([0.0336, 49.9206], nf);
feature_map = RFGProductEProdMap([0.0339, 50.4895], floor(sqrt(nf)));
%feature_map = RFGProductEProdMap([0.01, 1.0], floor(sqrt(nf)));
learner.opt('seed', seed);
learner.opt('feature_map', feature_map);
learner.opt('minibatch_size', length(trBundle));
learner.opt('max_gd_iter', 100);
learner.opt('reg_param', 1e-10);

s=learnMap(learner, trBundle, teBundle, bunName);
n=length(trBundle)+length(teBundle);
iden=sprintf('kllosslearner_%s_%s_%d.mat', class(learner), bunName, n);
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

