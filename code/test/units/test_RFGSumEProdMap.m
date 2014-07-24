function test_suite = test_RFGSumEProdMap()
    %
    initTestSuite;

end

function test_compareToExact()
    % compare to exact kernel evaluation 
    oldRng=rng();
    rng(11 );

    n=500;
    Means1=3*randn(1, n);
    Vars1=gamrnd(3, 4, 1, n);
    %fplot(@(x)gampdf(x, 3, 2), [0, 20])
    % set of Gaussian messages
    D1=DistNormal(Means1, Vars1);

    Means2=2*randn(1, n)+1;
    Vars2=gamrnd(4, 5, 1, n);
    D2=DistNormal(Means2, Vars2);

    % kernel parameter
    gwidth2s=[0.5, 5];

    ker1=KEGaussian(gwidth2s(1));
    ker2=KEGaussian(gwidth2s(2));
    K1=ker1.eval(D1, D1);
    K2=ker2.eval(D2, D2);
    % sum of kernels
    K=K1+K2;

    % number of random features
    numFeatures=1499;
    randMap=RFGSumEProdMap(gwidth2s, numFeatures);

    T=TensorInstances({DistArray(D1), DistArray(D2)});
    Z=randMap.genFeatures(T);
    assert(all(Z(:)>=-2 & Z(:)<=2));
    Kapprox= Z'*Z;
    Diff = abs(Kapprox-K);

    % plot kernels
    %figure
    %imagesc(K);
    %colorbar;
    %title('true kernel matrix');

    %figure 
    %imagesc(Kapprox);
    %colorbar
    %title('approx kernel matrix');

    display(sprintf('%s: mean abs diff: %.3f', mfilename, mean(Diff(:))));
    % should be much less than 1
    assert(mean(Diff(:))<1);

    dm=randMap.genFeaturesDynamic(T);
    dmZ=dm.toNumeric();
    assertVectorsAlmostEqual(dmZ, Z);

    g=randMap.getGenerator(T);
    I=600:800;
    J=50:100;
    assertVectorsAlmostEqual(g(I, J), Z(I, J));

    rng(oldRng);
end

