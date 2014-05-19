function test_suite = test_KMVGauss1()
%
initTestSuite;

end

function testKMVGauss1()
rng(3);
n = 100;
% generate data
s1.d = 1;
[M, V] = gen_mv(n);
s1.mean= M;
s1.variance = V;

s2.d=2;
[M, V] = gen_mv(n);
s2.mean= M;
s2.variance = V;

% initialize kernels
sigm = 1;
sigv = 3;

kfm = KGaussian(sigm);
kfv = KGaussian(sigv);
kf = KMVGauss1(sigm, sigv);

% test eval()
Kmat1 = kfm.eval(s1.mean, s2.mean).*kfv.eval(s1.variance, s2.variance);
Kmat2 = kf.eval(s1, s2);
assertVectorsAlmostEqual(Kmat1, Kmat2);

% test pairEval()
Kvec1 = kfm.pairEval(s1.mean, s2.mean).*kfv.pairEval(s1.variance, s2.variance);
Kvec2 = kf.pairEval(s1, s2);
assertVectorsAlmostEqual(Kvec1, Kvec2);

end

function testCandidates()
rng(4);
n = 100;
% generate data
s1.d = 1;
[M, V] = gen_mv(n);
s1.mean= M;
s1.variance = V;

C = KMVGauss1.candidates(s1, [1:2], [3:5]);
assert(length(C) == 6);

end

function [M,V]=gen_mv(n)
M = randn(1, n) + rand(1, n)*2;
V = 1+rand(1, n)*10;    
end