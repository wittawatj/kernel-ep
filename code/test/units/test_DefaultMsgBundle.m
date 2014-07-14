function test_suite = test_DefaultMsgBundle()
%
initTestSuite;

end

function test_basic()
    fakeParams=1:50;
    in1=DistNormal(fakeParams, fakeParams);
    in1a=DistArray(in1);
    in2=DistBeta(fakeParams, fakeParams);
    in2a=DistArray(in2);
    out=DistNormal(fakeParams, fakeParams);
    outa=DistArray(out);
    bundle=DefaultMsgBundle(outa, in1a, in2a);

    % splitTrainTest
    trProportion=0.8;
    [trBundle, teBundle]=bundle.splitTrainTest(trProportion);
    assert(trBundle.count()==40);
    assert(teBundle.count()==10);
    assert(bundle.count()==50);

end

function test_subsample()

    fakeParams=1:50;
    in1=DistNormal(fakeParams, fakeParams);
    in1a=DistArray(in1);
    in2=DistBeta(fakeParams, fakeParams);
    in2a=DistArray(in2);
    out=DistNormal(fakeParams, fakeParams);
    outa=DistArray(out);
    bundle=DefaultMsgBundle(outa, in1a, in2a);

    reducedBundle=bundle.subsample(20);
    assert(reducedBundle.count()==20);
    assert(bundle.count()==50);

end
