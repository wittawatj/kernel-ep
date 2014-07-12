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
    A = randn(6, 20);
    B = randn(20, 3);
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

