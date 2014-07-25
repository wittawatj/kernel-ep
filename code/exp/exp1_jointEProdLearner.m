%function [ ] = exp1_jointEProdLearner( )
%EXP1_JOINTEPRODLEARNER To test RFGJointEProdLearner and RFGJointEProdMap
%

seed=121+1;
oldRng=rng();
rng(seed);

se=BundleSerializer();
sample_cond_msg=false;
if sample_cond_msg
    anno='samcond';
else
    anno='proposal';
end
%bunName=sprintf('sigmoid_bw_%s_25000', anno);
%bunName=sprintf('sigmoid_bw_%s_2000', anno);
% Nicolas's data. Has almost 30000 pairs.
bunName=sprintf('nicolas_sigmoid_bw');
bundle=se.loadBundle(bunName);

%n=1000;
n=20000;
smallBundle=bundle.subsample(n);
n=min(n, smallBundle.count());
% train 80%
[trBundle, teBundle]=smallBundle.splitTrainTest(.8);

learner=RFGJointEProdLearner(trBundle);
od=learner.getOptionsDescription();
display(' Learner options: ');
od.show();
% set my options
%learner.opt('med_factors', [1]);
learner.opt('med_factors', [1/10, 1, 10]);
%learner.opt('num_primal_features', 2000);
learner.opt('num_primal_features', 10000);
learner.opt('candidate_primal_features', 2000);
%learner.opt('use_multicore', false);
learner.opt('use_multicore', true);
learner.opt('reglist', [1e-4, 1e-2, 1]);

% name used to identify the learned DistMapper for serialization purpose. 
%dmName=sprintf('learnDistMapper_%s_%d', bunName, n);
%dmSerializer=DistMapperSerializer();
if false
    display(sprintf('%s exists. Will load it.', dmName));
    % If exist, load and skip training.
    dm=dmSerializer.loadDistMapper(dmName);
else
    % learn a DistMapper
    [dm, Log]=learner.learnDistMapper();
    % save so it can be loaded next time
end

% test the learned DistMapper dm
% KL or Hellinger
tester=DivDistMapperTester(dm);
tester.opt('div_function', 'KL'); 
% test on the test MsgBundle
%keyboard
[Helling, outDa]=tester.testDistMapper(teBundle);
assert(isa(outDa, 'DistArray'));

% Check improper messages
impTester=ImproperDistMapperTester(dm);
impOut=impTester.testDistMapper(teBundle);


% save everything
expNumFolder=Expr.expSavedFolder(1);
iden=sprintf('%s_%s_%d.mat', class(learner), bunName, n);
fname=fullfile(expNumFolder, iden);
save(fname)
rng(oldRng);


%end

