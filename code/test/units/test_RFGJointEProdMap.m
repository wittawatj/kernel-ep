function test_suite = test_RFGJointEProdMap()
    %
    initTestSuite;

end

function test_tensorToJointGaussians()
    n=3;
    Means1=3*randn(2, n);
    Vars1=zeros(2, 2, n);
    for i=1:n
        Vars1(:, :, i)=wishrnd(eye(2), 5);
    end
    % set of Gaussian messages
    D1=DistNormal(Means1, Vars1);

    % set of Beta messages
    A=rand(1, n)*10+1;
    B=rand(1, n)*10+1;
    D2=DistBeta(A, B);
    T=TensorInstances({DistArray(D1), DistArray(D2)});
    D=RFGJointEProdMap.tensorToJointGaussians(T);

    assert(all([D.d]==3));
    assertVectorsAlmostEqual(D(2).variance(1:2, 1:2), Vars1(:,:,2));
    assertVectorsAlmostEqual(D(3).mean(1:2), Means1(:, 3));
end

function test_basic()
    % compare to exact kernel evaluation 
    oldRng=rng();
    rng(12 );

    n=200;
    Means1=3*randn(2, n);
    Vars1=zeros(2, 2, n);
    for i=1:n
        Vars1(:, :, i)=wishrnd(eye(2), 5);
    end
    % set of Gaussian messages
    D1=DistNormal(Means1, Vars1);

    % set of Beta messages
    A=rand(1, n)*10+1;
    B=rand(1, n)*10+1;
    D2=DistBeta(A, B);
    T=TensorInstances({DistArray(D1), DistArray(D2)});
    D=RFGJointEProdMap.tensorToJointGaussians(T);
    % kernel parameter
    gwidth2s=[2, 3];
    ker=KEGaussian( [gwidth2s(1), gwidth2s(1), gwidth2s(2)]);

    K=ker.eval(D, D);

    % number of random features
    numFeatures=799;
    randMap=RFGJointEProdMap(gwidth2s, numFeatures);

    Z=randMap.genFeatures(T);
    assert(all(Z(:)>=-1 & Z(:)<=1));
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

    g=randMap.getGenerator(T);
    I=300:500;
    J=50:100;
    assertVectorsAlmostEqual(g(I, J), Z(I, J));

    rng(oldRng);
end


