function  run_herdWeightsL1( )
% Experiment on weights herding with Lasso
%

seed=32;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
%bunName='sigmoid_bw_proposal_5000';
bunName='sigmoid_bw_proposal_2000';
%bunName='sigmoid_bw_proposal_10000';
%bunName='sigmoid_bw_proposal_20000';

%bunName='sigmoid_bw_fixbeta_10000';
%bunName='sigmoid_bw_fixbeta_10000';
bundle=se.loadBundle(bunName);

%[trBundle, teBundle] = bundle.partitionTrainTest(1000, 1000);
[trBundle, teBundle] = bundle.partitionTrainTest(500, 1000);
%[trBundle, teBundle] = bundle.partitionTrainTest(16000, 2000);

%Xtr = trBundle.getInputTensorInstances();
%Ytr = out_msg_distbuilder.getStat(trBundle.getOutBundle());
%out_msg_distbuilder = DNormalLogVarBuilder();
out_msg_distbuilder = DistNormalBuilder();
cond_factor = SigmoidCondFactor();
max_locations = 400;
lambdas = [1e-1];

learner = HerdLassoMapperLearner();
od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();

% set my options
learner.opt('seed', seed);
learner.opt('out_msg_distbuilder', out_msg_distbuilder);
learner.opt('cond_factor', cond_factor);
% set to x in sigmoid data. f(z|x). index=1 => z, index=2 => x.
learner.opt('target_index', 2);
learner.opt('max_locations', max_locations);
%learner.opt('num_lambda', 20);
learner.opt('lambdas', lambdas );
learner.opt('cv_fold', 2);

[dm, learnerLog] = learner.learnDistMapper(trBundle);

n=length(trBundle)+length(teBundle);
ntr = length(trBundle);
iden=sprintf('herd_weights_l1_%s_ntr%d_K%d.mat', bunName, ntr, max_locations);
fpath=Expr.scriptSavedFile(iden);

timeStamp=clock();
save(fpath, 'dm', 'learnerLog', 'timeStamp', 'trBundle', 'teBundle');

rng(oldRng);
end

