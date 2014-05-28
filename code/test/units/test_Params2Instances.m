function test_suite = test_Params2Instances()
%
initTestSuite;

end


function test_general()
    D(1) = DistNormal(1, 3);
    D(2) = DistNormal(2, 4);
    Ins = Params2Instances(D, 'mean', 'variance');
    
    % all fields are present
    S = Ins.getAll();
    fieldsPresent(S);
    
    % correct params
    assertVectorsAlmostEqual( S.param1, [1,2]);
    assertVectorsAlmostEqual( S.param2, [3,4]);
    
    % name
    assertEqual(S.param1Name, 'mean');
    assertEqual(S.param2Name, 'variance');
    
    assertEqual(Ins.count(), 2);
    
end

function fieldsPresent(S)
    assert(isfield(S, 'param1'));
    assert(isfield(S, 'param2'));
    assert(isfield(S, 'param1Name'));
    assert(isfield(S, 'param2Name'));
end