function test_suite = test_RFGProductEProdMap()
    %
    initTestSuite;

end

function test_compareToExact()
    % compare to exact kernel evaluation 
    oldRng=rng();
    rng(32 );

    n=100;
    Means1=3*randn(2, n);
    %Vars1=gamrnd(3, 4, 1, n);
    Vars1=zeros(2, 2, n);
    for i=1:n
        Vars1(:, :, i)=wishrnd(eye(2), 7);
    end
    %fplot(@(x)gampdf(x, 3, 2), [0, 20])
    % set of Gaussian messages
    D1=DistNormal(Means1, Vars1);

    Means2=2*randn(1, n)+1;
    Vars2=gamrnd(4, 5, 1, n);
    D2=DistNormal(Means2, Vars2);

    % kernel parameter
    gwidth2s=[1, 4];

    ker1=KEGaussian(gwidth2s(1)*ones(1, 2));
    ker2=KEGaussian(gwidth2s(2));
    K1=ker1.eval(D1, D1);
    K2=ker2.eval(D2, D2);
    % product of 2 kernels
    K=K1.*K2;

    % number of random features
    numFeatures=499;
    randMap=RFGProductEProdMap(gwidth2s, numFeatures);

    T=TensorInstances({DistArray(D1), DistArray(D2)});
    Z=randMap.genFeatures(T);
    Kapprox= Z'*Z;
    Diff = abs(Kapprox-K);
    RelDiff= abs( (Kapprox-K)./K );

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
    display(sprintf('%s: mean rel diff: %.3f', mfilename, mean(RelDiff(:))));
    % should be much less than 1
    assert(mean(Diff(:))<1);

    dm=randMap.genFeaturesDynamic(T);
    dmZ=dm.toNumeric();
    assertVectorsAlmostEqual(dmZ, Z);

    % test generator on a bunch of indices
    II={20:200, 100:300, 210:400 };
    JJ={2, 2:10, 50:100, 1:100};

    g=randMap.getGenerator(T);
    for i=1:length(II)
        I=II{i};
        J=JJ{i};
        assertVectorsAlmostEqual(g(I, J), Z(I, J));
    end

    rng(oldRng);
end

