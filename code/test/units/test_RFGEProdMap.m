function test_suite = test_RFGEProdMap()
    %
    initTestSuite;

end

function test_compareToExact()
    % compare to exact kernel evaluation 
    oldRng=rng();
    rng(10 );

    n=500;
    Means=3*randn(1, n);
    Vars=gamrnd(3, 4, 1, n);
    %fplot(@(x)gampdf(x, 3, 2), [0, 20])
    % set of Gaussian messages
    D=DistNormal(Means, Vars);
    % kernel parameter
    sigma2=3;
    ker=KEGaussian(sigma2);
    K=ker.eval(D, D);

    % number of random features
    numFeatures=1499;
    randMap=RFGEProdMap(sigma2, numFeatures);

    Z=randMap.genFeatures(D);
    assert(all(Z(:)>=-1 & Z(:)<=1));
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

    dm=randMap.genFeaturesDynamic(D);
    dmZ=dm.toNumeric();
    assertVectorsAlmostEqual(dmZ, Z);

    g=randMap.getGenerator(D);
    I=600:800;
    J=50:100;
    assertVectorsAlmostEqual(g(I, J), Z(I, J));
    
    rng(oldRng);
end

