function [  ] = exp6_logistic_factor(  )
%EXP6_LOGISTIC_FACTOR Experiment with logistic factor data.
%

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
%bunName = 'binlogis_fw_n400_iter5_sf1_st20';
%bunName = 'binlogis_bw_n400_iter5_sf1_st20';

bunName = 'binlogis_bw_proj_n400_iter5_sf1_st20';
%bunName = 'binlogis_fw_proj_n400_iter5_sf1_st20';
%
%bunName = 'binlogis_bw_n400_iter20_s1';
%bunName = 'binlogis_fw_n1000_iter5_s1';
%bunName='sigmoid_bw_proposal_10000';
%bunName='sigmoid_bw_proposal_5000';
%bunName='sigmoid_bw_proposal_1000';
%bunName = 'sigmoid_fw_proposal_5000';
bundle=se.loadBundle(bunName);

%n=5000;
%n=25000;
[trBundle, teBundle] = bundle.partitionTrainTest(5000, 2000);
%[trBundle, teBundle] = bundle.partitionTrainTest(100, 900);
%[trBundle, teBundle] = bundle.partitionTrainTest(3000, 1000);
%[trBundle, teBundle] = bundle.partitionTrainTest(6000, 4000);
%[trBundle, teBundle] = bundle.partitionTrainTest(30000, 5000);
%[trBundle, teBundle] = bundle.partitionTrainTest(500, 400);

%---------- options -----------
inTensor = trBundle.getInputTensorInstances();
% median factors 
medf = [1/20, 1/10, 1/5, 1/3, 1/2, 1, 2, 3, 5, 10, 20];
num_primal_features = 1000; 
num_inner_primal_features = 500;
%medf = [1];
%out_msg_distbuilder = DNormalSDBuilder();
%out_msg_distbuilder = DistNormalBuilder();
%out_msg_distbuilder = DistBetaBuilder();

out_msg_distbuilder = DNormalLogVarBuilder();
%out_msg_distbuilder = DBetaLogBuilder();

%kernel_choice = 'fm_kgg_joint';
kernel_choice = 'ichol_kgg_joint';
kernel_candidates = {};
fm_candidates = {};
if strcmp(kernel_choice, 'ichol_kgg_prod')
    kernel_candidates = KGGaussian.productCandidatesAvgCov(inTensor, medf, 2000);
    learner=ICholMapperLearner();

elseif strcmp(kernel_choice, 'ichol_kgg_sum')
    kernel_candidates = KGGaussian.ksumCandidatesAvgCov(inTensor, medf, 2000);
    learner=ICholMapperLearner();

elseif strcmp(kernel_choice, 'ichol_kgg_joint')
    kernel_candidates= KGGaussianJoint.candidatesAvgCov(inTensor, medf, 2000);
    learner=ICholMapperLearner();

elseif strcmp(kernel_choice, 'ichol_keg_prod')
    kernel_candidates = KEGaussian.productCandidatesAvgCov(inTensor, medf, 2000);
    learner=ICholMapperLearner();

elseif strcmp(kernel_choice, 'fm_kgg_joint')
    % random features for Gaussian kernel on mean embeddings on joint embeddings.
    %
    candidate_primal_features = 500;
    candidate_inner_primal_features = 300;

    fm_candidates = RFGJointKGG.candidatesAvgCov(inTensor, medf, ...
        candidate_inner_primal_features, candidate_primal_features, 2000);
    learner = RFGJointKGGLearner();
else 
    error(['invalid kernel_choice: ', kernel_choice]);
end

od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();
display(sprintf('Total %d kernel candidates for ichol.', length(kernel_candidates)));
display(sprintf('Total %d feature map candidates for random features.', length(fm_candidates)));

%
learner.opt('seed', seed);
learner.opt('out_msg_distbuilder', out_msg_distbuilder);
learner.opt('reglist', [1e-4, 1e-3, 1e-2, 1e-1, 1]);
learner.opt('use_multicore', true);
%learner.opt('use_multicore', false);
%
% ----- options for RFGJointKGGLearner ----
learner.opt('num_primal_features', num_primal_features);
learner.opt('num_inner_primal_features', num_inner_primal_features);
learner.opt('use_multicore', true);
learner.opt('featuremap_candidates', fm_candidates);
%
% --- options for ICholMapperLearner ----
learner.opt('kernel_candidates', kernel_candidates );
learner.opt('num_ho', 3);
learner.opt('ho_train_size', 1000);
learner.opt('ho_test_size', 1500);
%learner.opt('ho_train_size', 200);
%learner.opt('ho_test_size', 200);
%learner.opt('ho_train_size', 20);
%learner.opt('ho_test_size', 30);
learner.opt('chol_tol', 1e-8);
learner.opt('chol_maxrank_train', 100);
learner.opt('chol_maxrank', 800 );
learner.opt('separate_outputs', true);
%learner.opt('use_multicore', false);

%n=length(trBundle)+length(teBundle);
ntr = length(trBundle);
iden=sprintf('%s-irf%d-orf%d-%s-ntr%d-%s.mat', kernel_choice, num_inner_primal_features, ...
    num_primal_features, bunName, ntr, class(out_msg_distbuilder));
fpath=Expr.expSavedFile(6, iden);

s=learnMap(learner, trBundle, teBundle, bunName);
timeStamp=clock();
save(fpath, 's', 'timeStamp', 'trBundle', 'teBundle', 'out_msg_distbuilder');

rng(oldRng);



end

function s=learnMap(learner, trBundle, teBundle, bunName)
    % run the specified learner. 
    % Return a struct S containing produced variables.

    assert(isa(learner, 'DistMapperLearner'));
    assert(isa(trBundle, 'MsgBundle'));

    %n=length(trBundle)+length(teBundle);

    % learn a DistMapper
    [dm, learnerLog]=learner.learnDistMapper(trBundle);

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
    s.commit=commit;
    s.timeStamp=timeStamp;

end

