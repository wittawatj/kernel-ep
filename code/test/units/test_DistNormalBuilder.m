function test_suite = test_DistNormalBuilder()
%
initTestSuite;

end

function test_moments()
    m1=[0;1];
    v1=eye(2);
    D1=DistNormal(m1, v1);
    builder=DistNormalBuilder();
    Mcell=builder.getMoments(D1);
    assert(iscell(Mcell));
    assert(length(Mcell)==1);
    assert(length(Mcell{1})==2);
    assertVectorsAlmostEqual(Mcell{1}{1}, m1);
    assertVectorsAlmostEqual(Mcell{1}{2}-m1*m1', v1);

    
end

