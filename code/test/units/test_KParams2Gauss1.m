function test_suite = test_KParams2Gauss1()
%
initTestSuite;

end


function test_candidates()
    D(1) = DistNormal(1, 2);
    D(2) = DistNormal(5, 6);
    Ins = Params2Instances(D, 'mean', 'variance');
    s = Ins.getAll();
    
    p1_medf = [1, 2];
    p2_medf = [3, 4, 5];
    subsamples = 1000;
    
    Kcell = KParams2Gauss1.candidates(s, p1_medf, p2_medf, subsamples);
    assert(length(Kcell)== length(p1_medf)*length(p2_medf) );
    
end