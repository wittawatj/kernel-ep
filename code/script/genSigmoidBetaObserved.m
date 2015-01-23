% script to generate MsgBundle for sigmoid factor 
% Assume that the Bernoulli variable is observed
% Only two possible incoming messages from the variable: 
%   Beta(1, 2) or Beta(2, 1)
%
%n=5e4;
n=1e5;
% subsample into smaller pieces and save
%subsamples= [1e3, 2e4, 3e4, 4e4, 7e4, 1e5, 15e4, 2e5];
%subsamples = [2e4, 3e4, 4e4, 5e4];
subsamples = [6e4, 7e4, 8e4, 9e4, 1e5];

sg=SigmoidBundleGenerator();
% use proposal
sg.opt('seed', 4);
sg.opt('sample_cond_msg', false);
sg.opt('iw_samples', 8e4);
sg.opt('multicore_job_count', 7);
sg.opt('use_multicore', true);
sg.opt('is_beta_observed', true);
se=BundleSerializer();
fbundles=sg.genBundles(n);
fwBundle=fbundles.getMsgBundle(1);
bwBundle=fbundles.getMsgBundle(2);

for i=1:length(subsamples)
    subs=subsamples(i);

    anno='proposal';
    fwName=sprintf('sigmoid_fw_zobserved_%s_%d', anno, subs);
    bwName=sprintf('sigmoid_bw_zobserved_%s_%d', anno, subs);
    %fwName=sprintf('sigmoid_fw_fixbeta_%d',  subs);
    %bwName=sprintf('sigmoid_bw_fixbeta_%d',  subs);
    fwSmall=fwBundle.subsample(subs);
    bwSmall=bwBundle.subsample(subs);
    se.saveBundle(fwSmall, fwName);
    se.saveBundle(bwSmall, bwName);

end


