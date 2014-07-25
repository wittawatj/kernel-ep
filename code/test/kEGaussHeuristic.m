% This script investigates a good candidate for the 
% expected product kernel KEGaussian.
%

n=200;
Means=randn(1, n);
Vars=gamrnd(2, 4, 1, n);
D=DistNormal(Means, Vars);

ker0=KEGaussian(0.01);
K0=ker0.eval(D, D);

% median on means
mmed=meddistance(Means)^2;
vmed=meddistance(Vars);

width2=mmed/vmed;
ker=KEGaussian(width2);
K=ker.eval(D, D);

