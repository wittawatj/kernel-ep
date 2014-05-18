function test_suite = test_IncompChol()
%TEST_INCOMPCHOL Unit test for IncompChol
%
initTestSuite;

end

function s=setup()
% data for ichol
n = 200;
d = 30;

s.n = n;
s.d = d;
X = randn(d, n) + rand(d, n)*2;

med = meddistance(X);

kfunc = KGaussian(med^2);
s.Dat = MatInstances(X);

s.X = X;
s.med = med;
s.kfunc =kfunc;

tol = 1e-15;

% compare to unmodified pseudo-code from John Shawe Taylor book

% full K
K = kfunc.eval(X, X);
[fnewr1, R1, I1, nu1 ] = incomp_chol(K, tol);
s.fnewr1 = fnewr1;
s.R1 = R1;
s.I1 = I1;
s.nu1 = nu1;
s.K = K;
s.tol = tol;

end

function teardown(s)
end

function testExactIChol(s)
eval(structvars(1e2, s));

rng(2);

% maxcols = floor(n/4);
maxcols = n;
In=IncompChol(Dat, kfunc, tol, maxcols);
R2 = In.R;


% display(sprintf('Diff of two approximations: %g', norm(R1'*R1 - R2'*R2) ) );
assertVectorsAlmostEqual(R1'*R1, R2'*R2);

% test Cholesky projection
Y = randn(d, 100) + rand(d, 100)*2;
Rt2 = In.chol_project(MatInstances(Y));
Rt1 = fnewr1(kfunc.eval(X, Y));
% display(sprintf('Diff of two Chol. projections: %g', norm(Rt1'*Rt1-Rt2'*Rt2)));
assertVectorsAlmostEqual(Rt1'*Rt1, Rt2'*Rt2);

% diff to true matrix
% display(sprintf('Diff to full kernel matrix: %.g', norm(K-R2'*R2) ));
assert(norm(K-R2'*R2) < 1e-4, 'Exact Icholesky is too different from the full K');

end

function testIChol(s)
% test incomplete Cholesky with low rank
eval(structvars(1e2, s));

maxcols = 150;
In=IncompChol(Dat, kfunc, tol, maxcols);
R2 = In.R;

% display(sprintf('Diff of two approximations: %g', norm(R1'*R1 - R2'*R2) ) );
assert(norm(K - R2'*R2) < 10, 'Approximate IChol is too different from full K');



end
