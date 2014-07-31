% script to generate MsgBundle for simple Gaussian factor problem
%
oldRng=rng();
rng(910);

n=1e5;
%n=200;
% subsample into smaller pieces and save
subsamples=[2000, 1e4, 3e4, 5e4, 1e5];
%subsamples=[2e4, 5e4, 1e5];
%subsamples=[1e4, 25e3, 5e4 ];
%subsamples=[n];
%subsamples=[5e3];

sg=SimpleGaussBundleGenerator();
display(sg.shortSummary());
sample_cond_msg=false;
d=1;

sg.opt('sample_cond_msg', sample_cond_msg);
sg.opt('d', d);
% 1 million importance sampling weights
sg.opt('iw_samples', 1e6);
se=BundleSerializer();
fbundles=sg.genBundles(n);
fwBundle=fbundles.getMsgBundle(1);
bwBundle=fbundles.getMsgBundle(2);

if sample_cond_msg
    anno='samcond';
else
    anno='proposal';
end
for i=1:length(subsamples)
    subs=subsamples(i);

    fwName=sprintf('simplegauss_d%d_fw_%s_%d', d, anno, subs);
    bwName=sprintf('simplegauss_d%d_bw_%s_%d', d, anno, subs);
    fwSmall=fwBundle.subsample(subs);
    bwSmall=bwBundle.subsample(subs);
    se.saveBundle(fwSmall, fwName);
    se.saveBundle(bwSmall, bwName);

end

rng(oldRng);
