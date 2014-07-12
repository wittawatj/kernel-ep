n=1e5;
sg=SigmoidBundleGenerator();
se=BundleSerializer();
fbundles=sg.genBundles(n);
fwBundle=fbundles.getMsgBundle(1);
bwBundle=fbundles.getMsgBundle(2);

se.saveBundle(fwBundle, 'sigmoid_fw');
se.saveBundle(bwBundle, 'sigmoid_bw');

