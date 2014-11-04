function [ ] = ichol_learn_laplace( )
%ICHOL_LEARN_LAPLACE test learning operator with incomplete Cholesky using Laplace kernel
%

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
bunName='sigmoid_bw_proposal_10000';
%bunName='sigmoid_bw_proposal_50000_v6';
%bunName='sigmoid_bw_proposal_25000';
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
%[trBundle, teBundle] = bundle.partitionTrainTest(2000, 3000);
[trBundle, teBundle] = bundle.partitionTrainTest(6000, 4000);
%[trBundle, teBundle] = bundle.partitionTrainTest(10000, 10000);
%[trBundle, teBundle] = bundle.partitionTrainTest(40000, 10000);
%[trBundle, teBundle] = bundle.partitionTrainTest(500, 300);

%---------- options -----------
learner=ICholMapperLearner();
%inTensor = bundle.getInputTensorInstances();
% median factors 
medf = [1/10, 1/5, 1, 5, 10];
% make kernel candidates
zfe = MVParamExtractor();
%zfe = MLogVParamExtractor();
%zfe = NatParamExtractor();
xfe = NatParamExtractor();
%xfe = MVParamExtractor();
%xfe = MLogVParamExtractor();

kernel_candidates = getLaplaceKernelCandidates(bundle, zfe, xfe, medf);
display(sprintf('Totally %d kernel candidates for ichol.', length(kernel_candidates)));

od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();

% set my options
learner.opt('seed', seed);
learner.opt('out_msg_distbuilder', DNormalLogVarBuilder());
%learner.opt('out_msg_distbuilder', DNormalSDBuilder());
learner.opt('kernel_candidates', kernel_candidates );
learner.opt('num_ho', 3);
learner.opt('ho_train_size', 2000);
learner.opt('ho_test_size', 2000);
%learner.opt('ho_train_size', 200);
%learner.opt('ho_test_size', 200);
%learner.opt('chol_maxrank_train', 120);
learner.opt('chol_maxrank_train', 100);
learner.opt('chol_tol', 1e-15);
learner.opt('chol_maxrank', 2000);
learner.opt('use_multicore', true);
learner.opt('reglist', [1e-3, 1e-1, 1e-2, 1, 10]);
learner.opt('separate_outputs', true);

s=learnMap(learner, trBundle, teBundle, bunName);
n=length(trBundle)+length(teBundle);
iden=sprintf('ichol_learn_laplace_%s_%s_%d.mat', class(learner), bunName, n);
fpath=Expr.scriptSavedFile(iden);

timeStamp=clock();
save(fpath, 's', 'timeStamp', 'trBundle', 'teBundle');

rng(oldRng);
end

function candidates = getLaplaceKernelCandidates(bundle, zfe, xfe, medf)
    % z = Beta
    z = bundle.getInputBundle(1);
    x = bundle.getInputBundle(2);
    fz = zfe.extractFeatures(z);
    fx = xfe.extractFeatures(x);
    zmed = meddistance(fz);
    assert(zmed > 0);
    xmed = meddistance(fx);
    assert(xmed > 0);
    zkernels = arrayfun(@(w)KLaplace(w, zfe), zmed*medf, 'UniformOutput', false);
    xkernels = arrayfun(@(w)KLaplace(w, xfe), xmed*medf, 'UniformOutput', false);
    candidates = KProduct.cross_product(zkernels, xkernels);
    
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

