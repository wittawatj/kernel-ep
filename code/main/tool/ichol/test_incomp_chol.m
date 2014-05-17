% function  test_incomp_chol( input_args )
%TEST_INCOMP_CHOL test
% 
clear
seed = 4;
oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed', seed);
RandStream.setGlobalStream(rs);

% my OOP implementation 
n = 400;
d = 30;
tol = 1e-10;
% maxcols = floor(n/4);
maxcols = min(n, 399);
X = randn(d, n) + rand(d, n)*2;
med = meddistance(X);
kfunc = KGaussian(med^2);
Dat = MatInstances(X);
In=IncompChol(Dat, kfunc, tol, maxcols);
R2 = In.R;

% full K
K = kfunc.eval(X, X);
% [fnewr2, R2, I2, nu2 ] = incomp_chol2(K, tol);

% Diff to incomplete Cholesky approximation
imagesc( abs(K-R2'*R2));
colorbar

% compare to unmodified pseudo-code from John Shawe Taylor book
[fnewr1, R1, I1, nu1 ] = incomp_chol(K, tol);
display(sprintf('Diff of two approximations: %g', norm(R1'*R1 - R2'*R2) ) );

% test Cholesky projection
Y = randn(d, 100) + rand(d, 100)*2;
Rt2 = In.chol_project(MatInstances(Y));
Rt1 = fnewr1(kfunc.eval(X, Y));
display(sprintf('Diff of two Chol. projections: %g', norm(Rt1'*Rt1-Rt2'*Rt2)));

% diff to true matrix
display(sprintf('Diff to full kernel matrix: %.g', norm(K-R2'*R2) ));
RandStream.setGlobalStream(oldRs);

% end
