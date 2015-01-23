function [ ] = exp1_jointEProdLearner( )
%EXP1_JOINTEPRODLEARNER To test RFGJointEProdLearner and RFGJointEProdMap
%

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
bunName = 'sigmoid_bw_zobserved_proposal_10000';
%bunName='sigmoid_bw_proposal_10000';
%bunName='sigmoid_bw_proposal_1000';
%bunName=sprintf('sigmoid_bw_%s_25000', anno);
%bunName=sprintf('sigmoid_bw_%s_2000', anno);
% Nicolas's data. Has almost 30000 pairs.
%bunName=sprintf('nicolas_sigmoid_bw');
bundle=se.loadBundle(bunName);

ntr = 2000;
nte = 4000;
candidate_primal_features = 500;
use_cmaes = true;

medf=[1/100, 1/50, 1/10, 1/5, 1/3, 1/2, 1, 2, 3, 5, 10, 50, 100 ];

[trBundle, teBundle] = bundle.partitionTrainTest(ntr, nte);
trTensor=trBundle.getInputTensorInstances();

learner=RFGJointEProdLearner();
jointCandidates=RFGJointEProdMap.candidatesAvgCov(trTensor, medf, ...
    candidate_primal_features, 1000);
od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();
% set my options
learner.opt('seed', seed);
learner.opt('out_msg_distbuilder', DNormalLogVarBuilder());
%learner.opt('out_msg_distbuilder', DistNormalBuilder());
learner.opt('num_primal_features', 2000);
learner.opt('candidate_primal_features', candidate_primal_features);
%learner.opt('use_multicore', false);
learner.opt('featuremap_candidates', jointCandidates);
learner.opt('use_multicore', true);
%learner.opt('use_multicore', false);
learner.opt('use_cmaes', use_cmaes);
learner.opt('separate_outputs', true);
learner.opt('reglist', [1e-6, 1e-4, 1e-2, 1]);

opStruct = learner.options.opStruct;
opStruct = rmfield(opStruct, 'featuremap_candidates');

% name used to identify the learned DistMapper for serialization purpose. 
%dmName=sprintf('learnDistMapper_%s_%d', bunName, n);
%dmSerializer=DistMapperSerializer();

% learn a DistMapper
[dm, learnerLog]=learner.learnDistMapper(trBundle);
timeStamp = clock();
% save so it can be loaded next time

% save everything
cma_tags = {'', '_cma'};
cma_tag =  cma_tags{use_cmaes+1};
iden=sprintf('%s_%s_ntr%d%s.mat', class(learner), bunName, ntr, cma_tag);
expNumFile=Expr.expSavedFile(1, iden);
save(expNumFile, 'bunName', 'opStruct', 'trBundle', 'teBundle', 'dm', 'learnerLog', 'timeStamp');

rng(oldRng);


end

