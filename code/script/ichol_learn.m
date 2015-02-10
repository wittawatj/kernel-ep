function [ ] = ichol_learn( )
%ICHOL_LEARN test learning operator with incomplete Cholesky
%

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
%bunName = 'binlogis_fw_n400_iter5_sf1_st20';
bunName = 'binlogis_bw_n400_iter5_sf1_st20';
%bunName = 'binlogis_bw_n400_iter20_s1';
%bunName = 'binlogis_fw_n1000_iter5_s1';
%bunName='sigmoid_bw_proposal_10000';
%bunName = 'sigmoid_bw_zobserved_proposal_20000';
%bunName = 'sigmoid_bw_zobserved_proposal_40000';
%bunName='sigmoid_bw_proposal_5000';
%bunName = 'sigmoid_fw_proposal_5000';
% Nicolas's data. Has almost 30000 pairs.
%bunName=sprintf('nicolas_sigmoid_bw');
%bunName=sprintf('nicolas_sigmoid_fw');
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
%bunName=sprintf('simplegauss_d1_fw_proposal_30000' );
%bunName='lds_d3_tox_3000';
bundle=se.loadBundle(bunName);

%n=5000;
%n=25000;
[trBundle, teBundle] = bundle.partitionTrainTest(5000, 2000);
%[trBundle, teBundle] = bundle.partitionTrainTest(500, 1900);
%[trBundle, teBundle] = bundle.partitionTrainTest(3000, 1000);
%[trBundle, teBundle] = bundle.partitionTrainTest(6000, 4000);
%[trBundle, teBundle] = bundle.partitionTrainTest(30000, 5000);
%[trBundle, teBundle] = bundle.partitionTrainTest(500, 400);

%---------- options -----------
learner=ICholMapperLearner();
inTensor = bundle.getInputTensorInstances();
% median factors 
medf = [1/10, 1/5, 1/2, 1, 2, 5, 10];
%medf = [1];
%kernel_candidates=KEGaussian.productCandidatesAvgCov(inTensor, medf, 2000);
% in computing KGGaussian, non-Gaussian distributions will be treated as a Gaussian.
kernel_choice = 'prod_kegauss';

if strcmp(kernel_choice, 'prod')
    kernel_candidates = KGGaussian.productCandidatesAvgCov(inTensor, medf, 2000);
elseif strcmp(kernel_choice, 'sum')
    kernel_candidates = KGGaussian.ksumCandidatesAvgCov(inTensor, medf, 2000);
elseif strcmp(kernel_choice, 'joint')
    error('does not work because KGGaussian for multivariate case not implemented.');
    kernel_candidates= KGGaussianJoint.candidatesAvgCov(inTensor, medf, 2000);

elseif strcmp(kernel_choice, 'prod_kegauss')
    kernel_candidates = KEGaussian.productCandidatesAvgCov(inTensor, medf, 2000);
else 
    error(['invalid kernel_choice: ', kernel_choice]);
end

display(sprintf('Total %d kernel candidates for ichol.', length(kernel_candidates)));

od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();

%out_msg_distbuilder = DNormalSDBuilder();
%out_msg_distbuilder = DistNormalBuilder();
%out_msg_distbuilder = DistBetaBuilder();

out_msg_distbuilder = DNormalLogVarBuilder();
%out_msg_distbuilder = DBetaLogBuilder();
%
% set my options
learner.opt('seed', seed);
learner.opt('out_msg_distbuilder', out_msg_distbuilder);
learner.opt('kernel_candidates', kernel_candidates );
learner.opt('num_ho', 3);
learner.opt('ho_train_size', 1000);
learner.opt('ho_test_size', 1500);
%learner.opt('ho_train_size', 200);
%learner.opt('ho_test_size', 200);
learner.opt('chol_tol', 1e-8);
learner.opt('chol_maxrank_train', 100);
learner.opt('chol_maxrank', 800 );
learner.opt('reglist', [1e-4, 1e-3, 1e-2, 1e-1, 1]);
learner.opt('separate_outputs', true);
learner.opt('use_multicore', true);
%learner.opt('use_multicore', false);

s=learnMap(learner, trBundle, teBundle, bunName);
n=length(trBundle)+length(teBundle);
ntr = length(trBundle);
iden=sprintf('ichol_learn_%s_%s_%s_ntr%d_%s.mat', class(learner), bunName, kernel_choice, ntr, class(out_msg_distbuilder));
fpath=Expr.scriptSavedFile(iden);

timeStamp=clock();
save(fpath, 's', 'timeStamp', 'trBundle', 'teBundle', 'out_msg_distbuilder');

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

    %% test the learned DistMapper dm
    %% KL or Hellinger
    %divTester=DivDistMapperTester(dm);
    %divTester.opt('div_function', 'KL'); 
    %% test on the test MsgBundle
    %%keyboard
    %[Divs, outDa]=divTester.testDistMapper(teBundle);
    %assert(isa(outDa, 'DistArray'));

    %% Check improper messages
    %impTester=ImproperDistMapperTester(dm);
    %impOut=impTester.testDistMapper(teBundle);

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
    %s.div_tester=divTester;
    %s.divs=Divs;
    %s.out_distarray=outDa;
    %s.imp_tester=impTester;
    %s.imp_out=impOut;
    s.commit=commit;
    s.timeStamp=timeStamp;

end

