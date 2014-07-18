function test_suite = test_DefaultDynamicMatrix()
%
initTestSuite;

end

function test_mult()
    doMult(1);
    doMult(6);
    doMult(13);
    doMult(1e5);
end

function doMult(chunkSize)
    n=20;
    R=randi([2, 30], 1, n);
    C=randi([2, 30], 1, n);
    D=randi([2, 30], 1, n);
    for i=1:length(R)
        r=R(i);
        c=C(i);
        d=D(i);
        A = randn(r, c);
        B = randn(c, d);
        dma = DefaultDynamicMatrix.fromMatrix(A);
        dma.chunkSize = chunkSize;
        dmb = DefaultDynamicMatrix.fromMatrix(B);
        dmb.chunkSize = chunkSize;

        T = A*B;
        assertVectorsAlmostEqual(T, dmb.lmult(A) );
        assertVectorsAlmostEqual(T, dmb.lmult(dma) );
        assertVectorsAlmostEqual(T, dma.rmult(B) );
        assertVectorsAlmostEqual(T, dma.rmult(dmb) );
    end

end


function test_toNumeric()
    A = randn(6, 20);
    dm = DefaultDynamicMatrix.fromMatrix(A);
    assertVectorsAlmostEqual(A, dm.toNumeric() );
end

function test_mmt()
    A = randn(6, 20);
    dm = DefaultDynamicMatrix.fromMatrix(A);
    assertVectorsAlmostEqual(A*A', dm.mmt() );
end

function test_fromMatrix()
    A = randn(5, 10);
    [r, c] = size(A);
    dm = DefaultDynamicMatrix.fromMatrix(A);

    % test size()
    assert(r==size(dm, 1));
    assert(c==size(dm, 2));
    assert(all(size(A)==size(dm)));

    % check each element
    for i=1:r
        for j=1:c
            assertElementsAlmostEqual( A(i,j), dm.index(i,j) );
        end
    end

    % each column
    for j=1:c
        assertVectorsAlmostEqual( A(:, j), dm.col(j) );

    end

    % each row
    for i=1:r
        assertVectorsAlmostEqual( A(i, :), dm.row(i));
    end

end

function test_t()
    % test transpose 
    A = randn(10, 6);
    B = randn(10, 3);
    dma = DefaultDynamicMatrix.fromMatrix(A);
    dmb = DefaultDynamicMatrix.fromMatrix(B);

    assert(all(size(dma.t())==size(A') ));
    assertVectorsAlmostEqual(A', dma.t().toNumeric());
    assertVectorsAlmostEqual(B, dmb.t().t().toNumeric());

    assertElementsAlmostEqual(A(1,3), dma.t().index(3, 1));
    assertElementsAlmostEqual(B(5,2), dmb.t().t().index(5, 2));
    assertVectorsAlmostEqual(A(2, :), dma.t().col(2)');
    assertVectorsAlmostEqual(A(:, 3), dma.t().row(3)');
    assertVectorsAlmostEqual(A(2:5, 1:2), dma.t().index(1:2, 2:5)' );

    assertVectorsAlmostEqual(A'*B, dma.t().rmult(B));
    assertVectorsAlmostEqual(A'*B, dmb.lmult(dma.t()));

    C=randn(8,6);
    dmc=DefaultDynamicMatrix.fromMatrix(C);
    assertVectorsAlmostEqual(A*C', dma.rmult(dmc.t()));
    assertVectorsAlmostEqual(A*C', dmc.t().lmult(dma));

    dmct=dmc.t();
    %assertVectorsAlmostEqual(C'*C, dmc.t().rmult(C) );
end


