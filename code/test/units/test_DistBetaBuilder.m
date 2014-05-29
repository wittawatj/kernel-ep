function test_suite = test_DistBetaBuilder()
%
initTestSuite;

end

function test_stat()
    % gen many dists.
    n = 20;
    A = 1+rand(1, n)*20;
    B = 0.1+rand(1,n)*3;
    D = DistBeta(A, B);
    
    builder = DistBetaBuilder();
    S = builder.getStat(D);
    
    % check mean;
    assertVectorsAlmostEqual(S(1,:), [D.mean]);
    
    % check variance
    assertVectorsAlmostEqual(S(2,:), [D.variance]+[D.mean].^2);
    
    D2 = builder.fromStat(S);
    assertVectorsAlmostEqual([D.mean], [D2.mean]);
    assertVectorsAlmostEqual([D.variance], [D2.variance]);
    assertVectorsAlmostEqual([D.alpha], [D2.alpha]);
    assertVectorsAlmostEqual([D.beta], [D2.beta]);
    
end

function test_fromSamples()
    oldRng = rng();
    rng(2);
    
    d = DistBeta(3, 1);
    builder = DistBetaBuilder();
    
    N = 5e4;
    samples = d.sampling0(N);
    weights = ones(1, N)/N;
    d2 = builder.fromSamples(samples, weights);
    
    % may need to increase the threshold ?
    assert( abs(d.mean-d2.mean) < 1e-2 );
    assert( abs(d.variance-d2.variance) < 1e-2);
    assert( abs(d.alpha-d2.alpha) < 1e-1 );
    assert( abs(d.beta-d2.beta) < 1e-1);
    
    rng(oldRng);
end
