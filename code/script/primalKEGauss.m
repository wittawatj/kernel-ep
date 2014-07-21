function primalKEGauss()
    % Script to test the primal features for the expected product kernel 
    % (inner product of the mean embeddings of Gaussian messages using Gaussian 
    % kernel)
    %
    oldRng=rng();
    rng(10 );

    n=1e3;
    Means=3*randn(1, n);
    Vars=gamrnd(3, 4, 1, n);
    %fplot(@(x)gampdf(x, 3, 2), [0, 20])
    % set of Gaussian messages
    X=DistNormal(Means, Vars);
    % kernel parameter
    sigma2=3;
    ker=KEGaussian(sigma2);
    K=ker.eval(X, X);

    % repeat per D
    repeats=10;
    % try multiple D (#random features)
    Ds = [ 500:500:10000];
    %Ds=[500:500:1000];
    MaxErr = zeros(repeats, length(Ds)); 
    AvgErr = zeros(repeats, length(Ds));
    SDErr = zeros(repeats, length(Ds));

    % errors based on Frobenius norm
    %FroErr=zeros(1, length(Ds));

    for i=1:length(Ds)
        % number of random features
        D = Ds(i);

        for r=1:repeats
            % draw from Fourier transform of k. 
            % Width is the reciprocal of the original width
            W = randn(D, 1)/sqrt(sigma2);
            B = rand(D, 1)*2*pi;
            M=[X.mean];
            V=[X.variance];
            % cos terms
            C=cos(bsxfun(@plus, W*M, B));
            % exp terms
            E=exp(-0.5*(W.^2)*V);
            Z=sqrt(2/D)*C.*E; %Dxn
            Kran = Z'*Z;

            Diff = abs(Kran-K);
            AvgErr(r, i) = mean(Diff(:));
            MaxErr(r, i) = max(Diff(:));
            SDErr(r, i) = std(Diff(:));
            %FroErr(i)=sqrt(Diff(:)'*Diff(:));
        end
    end

    % plot
    hold all
    errorbar(Ds, mean(AvgErr), std(AvgErr, 0, 1), 'ro-', 'linewidth', 2);
    errorbar(Ds, mean(MaxErr), std(MaxErr, 0, 1), 'bo-', 'linewidth', 2);
    errorbar(Ds, mean(SDErr), std(SDErr, 0, 1), 'ko-', 'linewidth', 2);
    %plot(Ds, FroErr,  'linewidth', 2);
    set(gca, 'fontsize', 20);
    xlabel('#Random features');
    % ylabel('Norm of difference');
    ylabel('difference');
    title(sprintf('n=%d, repeats=%d', n, repeats));
    %legend('Avg diff.', 'max entry diff', 's.d. entry diff', 'Fro. diff');
    legend('Avg diff.', 'max entry diff', 's.d. entry diff' );
    grid on;
    hold off

    % imagesc(abs(Kran-K));
    % colorbar
    % keyboard;


    rng(oldRng);

end

function phiHat(X)
    assert(isa(X, 'DistNormal'));

end



