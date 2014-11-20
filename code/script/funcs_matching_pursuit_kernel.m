function  s=funcs_matching_pursuit_kernel( )
%FUNCS_MATCHING_PURSUIT_KERNEL functions for processing results of matching_pursuit_kernel
%   .

    s = struct();
    s.plotLearnedFunc = @plotLearnedFunc;
    s.getKernelFCCandidates = @getKernelFCCandidates;

end


function plotLearnedFunc()
    %fname = 'mp_laplace_sigmoid_bw_proposal_5000_5000.mat';
    %fname = 'mp_gauss_sigmoid_bw_proposal_10000_10000.mat';
    fname = 'mp_laplace_sigmoid_bw_proposal_10000_10000.mat';
    fpath = Expr.scriptSavedFile(fname);
    loaded = load(fpath, 'mp', 'trBundle', 'teBundle', 'out_msg_distbuilder');
    
    trBundle = loaded.trBundle;
    mp = loaded.mp;
    out_msg_distbuilder = loaded.out_msg_distbuilder;

    Xtr = trBundle.getInputTensorInstances();
    Ytr = out_msg_distbuilder.getStat(trBundle.getOutBundle());

    out1 = Ytr(1, :);
    f = mp.evalFunction(Xtr);

    % sort by true outputs.
    [sout, I] = sort(out1);
    sf = f(I);
    n = length(I);

    figure
    hold on;
    set(gca, 'fontsize', 20);
    plot(1:n, sout, 'ro-', 'LineWidth', 2);
    plot(1:n, sf,'bx-', 'LineWidth', 1 );
    legend('true output', 'MP learned function');
    hold off;
end

function candidates=getKernelFCCandidates(trBundle, zfe, xfe, medf)
    % z = Beta
    z = trBundle.getInputBundle(1);
    z = DistArray(z);
    x = trBundle.getInputBundle(2);
    x = DistArray(x);
    tensorTr = trBundle.getInputTensorInstances();
    n = length(x);
    % subsample
    subI = randperm(n, min(1e4, n));
    fz = zfe.extractFeatures(z(subI));
    fx = xfe.extractFeatures(x(subI));
    % for each feature dimension, find the median 
    zmeds = zeros(size(fz, 1), 1);
    xmeds = zeros(size(fx, 1), 1);
    for zi=1:length(zmeds)
        zmeds(zi) = meddistance(fz(zi, :));
    end
    for xi=1:length(xmeds)
        xmeds(xi) = meddistance(fx(xi, :));
    end
    zmeds = zmeds + rand(length(zmeds), 1)*0.1 + 1e-4;
    xmeds = xmeds + rand(length(xmeds), 1)*0.1 + 1e-4;
    assert(all(zmeds > 0));
    assert(all(xmeds > 0));
    % total number of candidats = len(medf)^totalDim
    allMeds = [zmeds(:)', xmeds(:)'];
    totalDim = length(allMeds);
    totalComb = length(medf)^totalDim;
    lap_candidates = cell(1, totalComb);
    gauss_candidates = cell(1, totalComb);

    % temporary vector containing indices
    % Total combinations can be huge ! Be careful. Exponential in the 
    % number of inputs
    I = cell(1, totalDim);
    fe = StackFeatureExtractor(zfe, xfe);
    centerInstances = tensorTr;
    centerFeatures = fe.extractFeatures(centerInstances);
    inputFeatures = fe.extractFeatures(tensorTr);
    for ci=1:totalComb
        [I{:}] = ind2sub( length(medf)*ones(1, totalDim), ci);
        II=cell2mat(I);
        kerWidths= medf(II).*allMeds ;
        kerzWidths = kerWidths(1:size(fz, 1));
        kerxWidths = kerWidths( (size(fz, 1)+1):end);
        widths = [kerzWidths(:); kerxWidths(:)];
        lap_candidates{ci} = KLaplaceFC(widths, fe, centerInstances, tensorTr, ...
            centerFeatures, inputFeatures);
        % need widths.^2 
        gauss_candidates{ci} = KGaussianFC(widths.^2, fe, centerInstances, tensorTr, ...
            centerFeatures, inputFeatures);

        % options
        mp_subsample = min(floor(0.8*n), 5000);
        mp_basis_subsample = min(length(centerInstances), 1000);
        lap_candidates{ci}.opt('mp_subsample', mp_subsample)
        lap_candidates{ci}.opt('mp_basis_subsample', mp_basis_subsample);
        gauss_candidates{ci}.opt('mp_subsample', mp_subsample)
        gauss_candidates{ci}.opt('mp_basis_subsample', mp_basis_subsample);
    end
    %candidates = [lap_candidates, gauss_candidates];
    %candidates = [lap_candidates ];
    candidates = [ gauss_candidates];


end

