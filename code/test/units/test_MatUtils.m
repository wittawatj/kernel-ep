function test_suite = test_MatUtils()
    %
    initTestSuite;

end

function test_colKronecker()

   % compare to Kronecker
   x=randn(10, 1);
   y=rand(10, 1)*5-2;
   assertVectorsAlmostEqual(kron(x, y), MatUtils.colKronecker(x, y));

   %
   X=[1, 2];
   Y=[1, 2; 3, 4];
   Z=[4 5; 6 7];
   XY=MatUtils.colKronecker(X, Y);
   assertVectorsAlmostEqual(XY, [1 4; 3 8]);
   XYZ=MatUtils.colKronecker(XY, Z);
   assertVectorsAlmostEqual(XYZ, [4, 20; 6, 28; 12 40; 18, 56]);

end

function test_quadraticFormAlong3Dim()
   d = 3;
   m = 10;
   n = 15;
   V = randn(d, m);
   Q = rand(d, d, n);
   vfunc = MatUtils.quadraticFormAlong3Dim(V, Q);
   vtrue = zeros(m, n);
   for i=1:m
       for j=1:n 
           vtrue(i, j) = V(:, i)'*Q(:, :, j)*V(:, i);
       end
   end
   assertVectorsAlmostEqual(vfunc, vtrue);

end

