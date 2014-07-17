% script to learn a DistMapper and test it with DivDistMapperTester 
%

seed=1e2+1;
oldRng=rng();
rng(seed);

se=BundleSerializer();
sample_cond_msg=false;
if sample_cond_msg
    anno='samcond';
else
    anno='proposal';
end
%bunName=sprintf('sigmoid_fw_%s_5000', anno);
% Nicolas's data. Has almost 50000 pairs.
bunName=sprintf('nicolas_sigmoid_fw');
bundle=se.loadBundle(bunName);

n=40000;
smallBundle=bundle.subsample(n);
n=min(n, smallBundle.count());
% train 70%
[trBundle, teBundle]=smallBundle.splitTrainTest(.8);

learner=RFGMVMapperLearner(trBundle);
od=learner.getOptionsDescription();
od.show();
% set my options
learner.opt('mean_med_factors', [1/5, 1, 5]);
learner.opt('variance_med_factors', [1/5, 1, 5]);
learner.opt('num_primal_features', 10000);
learner.opt('candidate_primal_features', 2000);
learner.opt('use_multicore', true);

% name used to identify the learned DistMapper for serialization purpose. 
dmName=sprintf('learnDistMapper_%s_%d', bunName, n);
dmSerializer=DistMapperSerializer();
if dmSerializer.exist(dmName)
    display(sprintf('%s exists. Will load it.', dmName));
    % If exist, load and skip training.
    dm=dmSerializer.loadDistMapper(dmName);
else
    % learn a DistMapper
    dm=learner.learnDistMapper();
    % save so it can be loaded next time
    dmSerializer.saveDistMapper(dm, dmName);
end

% test the learned DistMapper dm
% KL or Hellinger
tester=DivDistMapperTester(dm);
tester.opt('div_function', 'KL'); 
% test on the test MsgBundle
%keyboard
[Helling, outDa]=tester.testDistMapper(teBundle);
assert(isa(outDa, 'DistArray'));



rng(oldRng);

