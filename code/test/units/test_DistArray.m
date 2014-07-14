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

