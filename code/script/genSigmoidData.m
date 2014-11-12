% script to generate MsgBundle for sigmoid factor.
n=5e4;
% subsample into smaller pieces and save
%subsamples=[2e4, 5e4, 1e5];
%subsamples=[1e4, 25e3, 5e4 ];
%subsamples=[5e3];
%subsamples= [1e3, 2e4, 3e4, 4e4, 7e4, 1e5, 15e4, 2e5];
subsamples= [1e3, 2e3, 5e3, 1e4, 2e4, 5e4 ];
%subsamples= [1e3, 2e3, 5e3];
%subsamples = [2e4, 3e4, 5e4, 7e4, 1e5];

sg=SigmoidBundleGenerator();
% use proposal
sample_cond_msg=false;
sg.opt('seed', 4);
sg.opt('sample_cond_msg', sample_cond_msg);
sg.opt('iw_samples', 8e4);
sg.opt('multicore_job_count', 15);
sg.opt('use_multicore', false);
se=BundleSerializer();
fbundles=sg.genBundles(n);
fwBundle=fbundles.getMsgBundle(1);
bwBundle=fbundles.getMsgBundle(2);

for i=1:length(subsamples)
    subs=subsamples(i);

    if sample_cond_msg
        anno='samcond';
    else
        anno='proposal';
    end
    fwName=sprintf('sigmoid_fw_%s_%d', anno, subs);
    bwName=sprintf('sigmoid_bw_%s_%d', anno, subs);
    %fwName=sprintf('sigmoid_fw_fixbeta_%d',  subs);
    %bwName=sprintf('sigmoid_bw_fixbeta_%d',  subs);
    fwSmall=fwBundle.subsample(subs);
    bwSmall=bwBundle.subsample(subs);
    se.saveBundle(fwSmall, fwName);
    se.saveBundle(bwSmall, bwName);

end


