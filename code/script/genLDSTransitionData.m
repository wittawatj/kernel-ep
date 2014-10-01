% script to generate MsgBundle for LDS transition factor 
% See LDSTransitionBundleGenerator
% p(y|x,A) where y is the next time latent variable.
% x is the current-time latent variable.
%
rng(10);

%n=45e3;
n=4000;
% subsample into smaller pieces and save
%subsamples=[2e4, 5e4, 1e5];
%subsamples=[3000, 1e4, 2e4, 4e4 ];
%subsamples=[5, 10];
subsamples=[3000];

g=LDSTransitionBundleGenerator();
% use proposal
g.opt('iw_samples', 2e5);
d = 3;
g.opt('input_dim', d);
se=BundleSerializer();
% actual generation
fbundles=g.genBundles(n);

toYBundle=fbundles.getMsgBundle(1);
toXBundle=fbundles.getMsgBundle(2);
toABundle=fbundles.getMsgBundle(3);

for i=1:length(subsamples)
    subs=subsamples(i);

    yName=sprintf('lds_d%d_toy_%d', d, subs);
    xName=sprintf('lds_d%d_tox_%d', d, subs);
    aName=sprintf('lds_d%d_toa_%d', d, subs);

    se.saveBundle(toYBundle.subsample(subs), yName);
    se.saveBundle(toXBundle.subsample(subs), xName);
    se.saveBundle(toABundle.subsample(subs), aName);

end



