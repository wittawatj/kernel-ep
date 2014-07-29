function test_suite = test_MatUtils()
    %
    initTestSuite;

end

function test_colOutputProduct()
    % compare to Kronecker
    x=randn(10, 1);
    y=rand(10, 1)*5-2;
    assertVectorsAlmostEqual(kron(x, y), MatUtils.colOutputProduct(x, y));

    %
    X=[1, 2];
    Y=[1, 2; 3, 4];
    Z=[4 5; 6 7];
    XY=MatUtils.colOutputProduct(X, Y);
    assertVectorsAlmostEqual(XY, [1 4; 3 8]);
    XYZ=MatUtils.colOutputProduct(XY, Z);
    assertVectorsAlmostEqual(XYZ, [4, 20; 6, 28; 12 40; 18, 56]);

end

