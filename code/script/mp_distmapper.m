function  mp_distmapper( )
%MP_DISTMAPPER Test MPMapperLearner .
%   .

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
%bunName='sigmoid_bw_proposal_5000';
%bunName='sigmoid_bw_proposal_2000';
bunName='sigmoid_bw_proposal_10000';
%bunName='sigmoid_bw_proposal_20000';
%bunName='sigmoid_bw_proposal_50000';

%bunName='sigmoid_bw_fixbeta_10000';
%bunName='sigmoid_bw_fixbeta_10000';
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
%bunName=sprintf('simplegauss_d1_fw_proposal_30000' );
bundle=se.loadBundle(bunName);

%[trBundle, teBundle] = bundle.partitionTrainTest(2000, 2000);
[trBundle, teBundle] = bundle.partitionTrainTest(8000, 2000);
%[trBundle, teBundle] = bundle.partitionTrainTest(15000, 2000);

%[trBundle, teBundle] = bundle.partitionTrainTest(1000, 1000);

%[trBundle, teBundle] = bundle.partitionTrainTest(4000, 1000);

%---------- options -----------
Xtr = trBundle.getInputTensorInstances();
%Ytr = out_msg_distbuilder.getStat(trBundle.getOutBundle());
out_msg_distbuilder = DNormalLogVarBuilder();

% median factors 
medf = [1/4, 1, 4 ];
%medf = [1/20, 1/10, 1/5, 1/2, 1, 2, 5, 10, 20 ];

mp_reg = 1e-2;
zfe = MVParamExtractor();
%xfe = NatParamExtractor();
xfe = MLogVParamExtractor();
% in order of variables p(z | x)
s = funcs_matching_pursuit_kernel();
feature_candidates = s.getKernelFeatureFCCandidates(trBundle, zfe, xfe, medf);
ker_candidates = s.getKernelFCCandidates(Xtr, medf);
fc_candidates = [feature_candidates(:)', ker_candidates(:)'];
%fc_candidates = s.getKernelFCLinearCandidates(trBundle, zfe, xfe, medf);
% limit fc_candidates 
c = length(fc_candidates);
J = randperm(c, min(c, 5e3));
fc_candidates = fc_candidates(J);
display(sprintf('Totally %d function class candidates.', length(fc_candidates)));

% set my options
opt = struct();
opt.seed = seed;
opt.mp_function_classes = fc_candidates;
opt.mp_reg = mp_reg;
opt.mp_max_iters = 100;
opt.mp_backfit_every = 1;
opt.mp_fc_subset = 100;

% start matching pursuit
learner = MPMapperLearner();
learner.opt('seed', seed);
learner.opt('out_msg_distbuilder', out_msg_distbuilder);
learner.opt('separate_outputs', true);
learner.opt('mp_options', opt);
[dm, learnerLog] = learner.learnDistMapper(trBundle);
% mp_options contains all function class candidates (huge). Exclude so that 
% the saved file is small.
learner.opt('mp_options', []);

%n=length(trBundle)+length(teBundle);
ntr = length(trBundle);
iden=sprintf('mp_distmapper_%s_ntr%d.mat',  bunName, ntr);
%iden=sprintf('mp_distmapper_linearcan_%s_ntr%d.mat',  bunName, ntr);

fpath=Expr.scriptSavedFile(iden);

timeStamp=clock();
save(fpath, 'dm', 'learnerLog', 'learner', 'out_msg_distbuilder', 'timeStamp', 'trBundle', 'teBundle');


rng(oldRng);
end

