function test_suite = test_DistBeta()
%
initTestSuite;

end

function test_mean()
    d=DistBeta(1, 4);
    assertElementsAlmostEqual(d.mean, 1/5 );
end


function test_variance()
    d=DistBeta(1, 4);
    assertElementsAlmostEqual(d.variance, 2/75);
end

function test_draw()
    d=DistBeta(2,5);
    s=d.draw(50);
    assertTrue(all(s>=0) );
    assertTrue(all(s<=1) );
end

function test_isproper()
    assert(DistBeta(2, 3).isproper());
    assert(~DistBeta(0, 3).isproper());
    assert(~DistBeta(2, 0).isproper());
    assert(~DistBeta(inf, 3).isproper());
    assert(~DistBeta(2, nan).isproper());
    
end

function test_parameters()
    a = 4;
    b = 1;
    d = DistBeta(a, b);
    C = d.parameters;
    assertElementsAlmostEqual(C{1}, a);
    assertElementsAlmostEqual(C{2}, b);
end