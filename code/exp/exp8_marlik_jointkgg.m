function [  ] = exp8_marlik_jointkgg(  )
%EXP8_MARLIK_JOINTKGG Experiment the parameter selection by gradient ascent 
%on the marginal likelihood using Gaussian kernel of joint mean embeddings.
%

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
%bunName = 'binlogis_fw_n400_iter5_sf1_st20';
%bunName = 'binlogis_bw_n400_iter5_sf1_st20';

bunName = 'binlogis_bw_proj_n400_iter5_sf1_st20';
%bunName = 'sigmoid_bw_proposal_1000';
%bunName = 'binlogis_fw_proj_n400_iter5_sf1_st20';
%
%bunName = 'binlogis_bw_n400_iter20_s1';
%bunName = 'binlogis_fw_n1000_iter5_s1';
bundle=se.loadBundle(bunName);

%n=5000;
%n=25000;
%[trBundle, teBundle] = bundle.partitionTrainTest(1000, 3000);
[trBundle, teBundle] = bundle.partitionTrainTest(1000, 3000);
%[trBundle, teBundle] = bundle.partitionTrainTest(100, 900);
%[trBundle, teBundle] = bundle.partitionTrainTest(3000, 1000);

%---------- options -----------
inTensor = trBundle.getInputTensorInstances();
%out_msg_distbuilder = DistBetaBuilder();

out_msg_distbuilder = DNormalLogVarBuilder();
%out_msg_distbuilder = DBetaLogBuilder();

learner = RFGJointKGGGrad2MLLearner();

od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();

medf = [1];
in_features = 500;
out_features = 1000;
fm_candidates = RFGJointKGG.candidatesAvgCov(inTensor, medf, ...
    in_features, out_features, 2000);
% Use the median and average covariance heuristics to initialize the marginal
% likelihood optimization
fm = fm_candidates{1};
init_outer_width2 = fm.outer_width2;
init_embed_width2s = MatUtils.flattenCell(fm.embed_width2s_cell);
%
learner.opt('seed', seed);
learner.opt('out_msg_distbuilder', out_msg_distbuilder);
learner.opt('prior_var', 1);
% noise_var is roughtly the same as a regularization parameter in ridge regression.
learner.opt('init_noise_var', 1e-4);
learner.opt('init_outer_width2', init_outer_width2);
learner.opt('init_embed_width2s', init_embed_width2s);
learner.opt('num_inner_primal_features', in_features);
learner.opt('num_primal_features', out_features);
%
%

%n=length(trBundle)+length(teBundle);
ntr = length(trBundle);
iden=sprintf('marlik-irf%d-orf%d-%s-ntr%d-%s.mat', in_features, ...
    out_features, bunName, ntr, class(out_msg_distbuilder));
fpath=Expr.expSavedFile(8, iden);

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

