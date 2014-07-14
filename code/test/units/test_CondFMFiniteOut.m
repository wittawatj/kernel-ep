function test_suite = test_CondFMFiniteOut()
%
initTestSuite;

end

function test_saveload()

    mwidth2s=[1];
    vwidth2s=[2];
    numFeatures=100;
    randFeatureMap=RandFourierGaussMVMap(mwidth2s, vwidth2s, numFeatures);

    means=1:10;
    In=DistArray(DistNormal(means, 1:10) );
    Out=randn(3, 10);
    lamb=15;
    condFm=CondFMFiniteOut(randFeatureMap, In, Out, lamb);

    % save
    fname='test_CondFMFiniteOut_saveload.mat';
    save(fname, 'condFm');
    oldFm=condFm;

    clear condFm
    %load
    load(fname);
    assertVectorsAlmostEqual(oldFm.mapMatrix, condFm.mapMatrix);
    assertElementsAlmostEqual(oldFm.regParam, condFm.regParam);
    assertVectorsAlmostEqual(oldFm.featureMap.rfgMap.W, condFm.featureMap.rfgMap.W);
    assertVectorsAlmostEqual(oldFm.featureMap.rfgMap.B, condFm.featureMap.rfgMap.B);


    delete(fname);

end
