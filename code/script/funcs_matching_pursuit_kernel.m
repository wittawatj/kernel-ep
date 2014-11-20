function  s=funcs_matching_pursuit_kernel( )
%FUNCS_MATCHING_PURSUIT_KERNEL functions for processing results of matching_pursuit_kernel
%   .

    s = struct();
    s.plotLearnedFunc = @plotLearnedFunc;

end


function plotLearnedFunc()
    fname = 'mp_laplace_sigmoid_bw_proposal_5000_5000.mat';
    fpath = Expr.scriptSavedFile(fname);
    loaded = load(fpath, 'mp', 'trBundle', 'teBundle');
    
    trBundle = loaded.trBundle;
    mp = loaded.mp;

    Xtr = trBundle.getInputTensorInstances();
    out_msg_distbuilder = DNormalLogVarBuilder();
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
