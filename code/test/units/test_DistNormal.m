function test_suite = test_DistNormal()
%
initTestSuite;

end

function test_distHellinger()

rng(2);

% test bound [0,1], symmetry
for i=1:50
    d1 = DistNormal(randn(1)*30, rand(1)*20);
    % self distance = 0
    assertElementsAlmostEqual(d1.distHellinger(d1), 0 );

    d2 = DistNormal(randn(1)*2, rand(1)*50 );
    dist1 = d1.distHellinger(d2);
    dist2 = d2.distHellinger(d1);
    % bound
    assertTrue(dist1>=0);
    assertTrue(dist1<=1);
    
    % symmetry
    assertElementsAlmostEqual(dist1, dist2);
end


end

function test_getMean()
    d=DistNormal(1, 4);
    assertElementsAlmostEqual(d.getMean(), 1);
end


function test_getVariance()
    d=DistNormal(1, 4);
    assertElementsAlmostEqual(d.getVariance(), 4);
end