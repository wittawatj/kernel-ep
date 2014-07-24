function test_suite = test_DistArray()
%
initTestSuite;

end

function test_basic()
    fakeParams=1:20;
    in1=DistNormal(fakeParams, 2*fakeParams);
    in1a=DistArray(in1);
    E=cellfun(@eq, in1a.getParamNames(), {'mean', 'variance'}, 'UniformOutput', false) ;
    % make sure that getParamNames() on DistArray returns {'mean', 'variance'}
    % for DistNormal distributions
    assert(all([E{:}]));

end

function test_subsref()
    D=DistNormal(randn(2, 5), reshape(1:20, [2,2,5]));
    da=DistArray(D);

    d2=da(2);
    % DistArray does not support da(2).mean. 
    assertVectorsAlmostEqual(d2.mean, D(2).mean);
    assertVectorsAlmostEqual(d2.variance, D(2).variance);

    d23=da(2:3);
    assertVectorsAlmostEqual([d23.mean], [D(2:3).mean]);
    assertVectorsAlmostEqual(cat(3, d23.variance), cat(3, D(2:3).variance));

end

