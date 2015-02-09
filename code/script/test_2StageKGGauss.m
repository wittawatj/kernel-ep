%function test_2StageKGGauss()
    % Script to test two-stage random features for the Gaussian kernel on mean 
    % embeddings. The embedding kernel is a Gaussian.
    %
    
    % dataset (messages)
    seed = 1; 
    oldRng=rng();
    rng(seed );

    n=300;
    Means=randn(1, n);
    Vars=gamrnd(2, 4, 1, n);
    %fplot(@(x)gampdf(x, 3, 2), [0, 20])
    % set of Gaussian messages
    X=DistNormal(Means, Vars);
    % kernel parameter
    %
    %g
    outer_width2 = 0.2;
    inner_embed_width2 = 2;

    ker=KGGaussian(inner_embed_width2, outer_width2);
    % true kernel matrix
    K=ker.eval(X, X);

    % repeat per D
    repeats=20;
    % try multiple D (#random features). 
    % We have inner and outer numbers of random features
    Dinners = 100:50:700 ;
    Douters = 100:50:700 ;

    %Ds=[500:500:1000];
    % errors based on Frobenius norm
    FroErr = zeros(repeats, length(Dinners), length(Douters));
    MaxErr = zeros(repeats, length(Dinners), length(Douters));

    for i=1:length(Dinners)
        for j=1:length(Douters)
            % number of random features
            Din = Dinners(i);
            Dout = Douters(j);

            for r=1:repeats
                % draw from Fourier transform of k. 
                % Width is the reciprocal of the original width
                W_in = randn(Din, 1)/sqrt(inner_embed_width2);
                B = rand(Din, 1)*2*pi;
                M=[X.mean];
                V=[X.variance];
                % cos terms
                C=cos(bsxfun(@plus, W_in*M, B));
                % exp terms
                E=exp(-0.5*(W_in.^2)*V);
                Phi=sqrt(2/Din)*C.*E; %Din x n

                % Phi contain random features for the inner Gaussian. 
                % We then approximate the outer kernel by treating Phi as 
                % data in a Euclidean Gaussian kernel. 
                Nu_out = randn(Dout, Din)/sqrt(outer_width2);
                C = rand(Dout, 1)*2*pi;
                % Dout x n
                Psi = sqrt(2/Dout)*cos(bsxfun(@plus, Nu_out*Phi, C));
                Kran = Psi'*Psi;

                Diff2 = (Kran-K).^2;
                FroErr(r, i, j) = sqrt(sum(Diff2(:)));
                MaxErr(r, i, j) = max(max(max(Kran, K)));
                %FroErr(i)=sqrt(Diff(:)'*Diff(:));
                fprintf('Din: %d, Dout: %d, repeat: %d \n', Din, Dout, r);
            end
        end
    end

    AvgFroErr = shiftdim(mean(FroErr, 1), 1);
    SDFroErr = shiftdim(std(FroErr, [], 1), 1);
    AvgMaxErr = shiftdim(mean(MaxErr, 1), 1);
    SDMaxErr = shiftdim(std(MaxErr, [], 1), 1);
    DinnersStr = arrayfun(@(x)(sprintf('%d', x)), Dinners, 'UniformOutput', false);
    DoutersStr = arrayfun(@(x)(sprintf('%d', x)), Douters, 'UniformOutput', false);
    % plot
    hold all
    [X, Y] = meshgrid(Dinners, Douters);
    %contourf(X, Y, AvgMaxErr);
    contourf(X, Y, AvgFroErr);
    set(gca, 'XLim', [min(Dinners), max(Dinners)]);
    set(gca, 'XTick', Dinners)
    set(gca, 'XTickLabel', DinnersStr);
    set(gca, 'YLim', [min(Douters), max(Douters)]);
    set(gca, 'YTick', Douters', 'YTickLabel', DoutersStr);
    set(gca, 'fontsize', 20);
    title(sprintf('n=%d, repeats=%d', n, repeats));
    %legend('Avg diff.', 'max entry diff', 's.d. entry diff' );
    xlabel('inner #features');
    ylabel('outer #features');
    grid on;
    hold off
    colorbar

    % imagesc(abs(Kran-K));
    % colorbar
    % keyboard;


    rng(oldRng);

%end
