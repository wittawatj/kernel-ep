function [ s] = funcs_test_cg(  )
%FUNCS_TEST_CG A collection of functions to test compound Gamma function
%

    s = struct();
    s.drawGaussianCGPrior = @drawGaussianCGPrior;
    s.drawGamShapeRate = @drawGamShapeRate;
    s.drawCompoundGam = @drawCompoundGam;

end

function sam = drawGamShapeRate(s, r, m, n)
    sam = gamrnd(s, 1/r, m, n);
end

function precs = drawCompoundGam(s1, r1, s2, n)
    R2 = drawGamShapeRate(s1, r1, 1, n);
    precs = gamrnd(s2, 1./R2);

end

function x = drawGaussianCGPrior(s1, r1, s2, mu, n)
    % Drawn n samples from a Gaussian with compound Gamma prior on its precision.
    % s1 = shape1, r1= rate1, s2= shape2, mu = mean of the Gaussian 
    %

    precs = drawCompoundGam(s1, r1, s2, n);
    x = randn(1, n).*(1./sqrt(precs)) + mu;
end

