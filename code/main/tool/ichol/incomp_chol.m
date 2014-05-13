function [fnewr, R, I, nu ] = incomp_chol(K, eta)
% Incomplete Cholesky decomposition
% - kernel matrix K (ell x ell)
% - eta gives threshold residual cutoff
% Return R = new features stored in matrix R of size T x ell
%

assert(all(size(K)==size(K')), 'K must be square');
assert(eta>=0, 'Threshold residual cutoff must be non-negative');
ell = size(K,1);

j = 0;
R = zeros(ell, ell);
d = diag(K);
[a, I(j+1)] = max(d);
if a <= eta
    error('eta too high. Should be: eta < max(diag(K)).');
end

while a > eta
    j = j+1;
    nu(j) = sqrt(a);
    for i=1:ell
        R(j, i) = (K(I(j), i) - R(:,i)'*R(:,I(j)))/nu(j);
        d(i) = d(i) - R(j,i)^2;
    end
    [a, I(j+1)] = max(d);
end
T = j;
R = R(1:T, :);
I = I(1:T);
fnewr = @(Ktest)(newr(Ktest, R, I, nu));


end



function Rt=newr(Ktest, R, I, nu)
% for new example with vector of inner products
% Ktest of size ell x m to compute new features r
% Return R of size T x m

[n,M] = size(Ktest);
T = size(R,1);
Rt = zeros(T, M);
for m=1:M
    r = zeros(T, 1);
    k = Ktest(:, m);
    for j=1:T
        r(j) = (k(I(j)) -r'*R(:,I(j)))/nu(j);
    end
    Rt(:, m) = r;
end

end
