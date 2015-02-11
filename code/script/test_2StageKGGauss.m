%function test_2StageKGGauss()
    % Script to test two-stage random features for the Gaussian kernel on mean 
    % embeddings. The embedding kernel is a Gaussian.
    %
    
    % dataset (messages)
    seed = 1; 
    oldRng=rng();
    rng(seed );

    n=300;
    % 2d Gaussians
    dim = 3;
    Means=randn(dim, n);
    Vars = zeros(dim, dim, n);
    %Vars=gamrnd(2, 4, 1, n);
    for i=1:n
        Vars(:, :, i) = wishrnd(eye(dim), 5);
    end
    %fplot(@(x)gampdf(x, 3, 2), [0, 20])
    % set of Gaussian messages
    X=DistNormal(Means, Vars);
    % kernel parameter
    %
    %g
    outer_width2 = 1;
    inner_embed_width2 = 2*ones(1, dim);

    ker=KGGaussian(inner_embed_width2, outer_width2);
    % true kernel matrix
    K=ker.eval(X, X);

    % repeat per D
    repeats=50;
    % try multiple D (#random features). 
    % We have inner and outer numbers of random features
    Dinners = 100:50:600 ;
    Douters = 100:50:700 ;

    %Ds=[500:500:1000];
    % errors based on Frobenius norm
    FroErr = zeros( length(Dinners), length(Douters), repeats);
    MaxErr = zeros( length(Dinners), length(Douters), repeats);

    for i=1:length(Dinners)
        for j=1:length(Douters)
            % number of random features
            Din = Dinners(i);
            Dout = Douters(j);

            for r=1:repeats
                % draw from Fourier transform of k. 
                % Width is the reciprocal of the original width
                W_in = randn(Din, dim)*diag(1./sqrt(inner_embed_width2));
                B = rand(Din, 1)*2*pi;
                M=[X.mean];
                V=cat(3, X.variance);
                % cos terms
                Cos=cos(bsxfun(@plus, W_in*M, B));
                % exp terms
                E = exp(-0.5*MatUtils.quadraticFormAlong3Dim(W_in', V)); % Din x n
                %E=exp(-0.5*(W_in.^2)*V);
                Phi=sqrt(2/Din)*Cos.*E; %Din x n

                % Phi contain random features for the inner Gaussian. 
                % We then approximate the outer kernel by treating Phi as 
                % data in a Euclidean Gaussian kernel. 
                Nu_out = randn(Dout, Din)/sqrt(outer_width2);
                C = rand(Dout, 1)*2*pi;
                % Dout x n
                Psi = sqrt(2/Dout)*cos(bsxfun(@plus, Nu_out*Phi, C));
                Kran = Psi'*Psi;

                Diff2 = (Kran-K).^2;
                FroErr(i, j, r) = sqrt(sum(Diff2(:)));
                MaxErr(i, j, r) = max(max(max(Kran, K)));
                %FroErr(i)=sqrt(Diff(:)'*Diff(:));
                fprintf('Din: %d, Dout: %d, repeat: %d \n', Din, Dout, r);
            end
        end
    end

    AvgFroErr = mean(FroErr, 3);
    SDFroErr = std(FroErr, [], 3);
    AvgMaxErr = mean(MaxErr, 3);
    SDMaxErr = std(MaxErr, [], 3);
    DinnersStr = arrayfun(@(x)(sprintf('%d', x)), Dinners, 'UniformOutput', false);
    DoutersStr = arrayfun(@(x)(sprintf('%d', x)), Douters, 'UniformOutput', false);
    % plot
    hold all
    %X = repmat(Dinners(:), 1, length(Douters));
    %Y = repmat(Douters(:)', length(Dinners), 1);
    [X, Y] = meshgrid(Douters, Dinners);
    contourf(X, Y, AvgFroErr/n);
    %contourf(Douters, Dinners, AvgFroErr/n);
    set(gca, 'YLim', [min(Dinners), max(Dinners)]);
    set(gca, 'YTick', Dinners)
    set(gca, 'YTickLabel', DinnersStr);
    set(gca, 'XLim', [min(Douters), max(Douters)]);
    set(gca, 'XTick', Douters', 'YTickLabel', DoutersStr);
    set(gca, 'fontsize', 20);
    title(sprintf('n=%d, repeats=%d. Report Fro. norm(K - Khat) / n.', n, repeats));
    %legend('Avg diff.', 'max entry diff', 's.d. entry diff' );
    ylabel('inner #features');
    xlabel('outer #features');
    grid on;
    hold off
    colorbar

    % imagesc(abs(Kran-K));
    % colorbar
    % keyboard;

    save('test_2StageKGGauss')

    rng(oldRng);

%end
