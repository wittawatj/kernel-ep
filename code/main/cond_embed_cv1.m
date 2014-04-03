function CVLog = cond_embed_cv1(X, Z,  op)
%
% Cross validation for conditional mean embedding for one conditioning
% variable. The embedding operator takes the form C_{z|x} where X is the
% conditioning variable.
% - Assume Gaussian kernels.
% - All data matrices are dim x dataset size
% The output mapping for Z represents sufficient statistic for a normal
% distribution, [z, z^2] (finite-dimensional mapping, no parameters).
%

[d,n] = size(X);

if nargin < 3
    op = [];
end
% list of regularization parameters. Should some how depend on n ??
reglist = myProcessOptions(op, 'reglist', [1e-2, 1e-0, 10]);
% list of Gaussian widths for X. This is to be multiplied with pairwise
% median distance.
list = [1, 2, 4];
xwlist = myProcessOptions(op, 'xwlist', list);

medx = meddistance(X);

% Number of folds in cross validation
fold = myProcessOptions(op, 'fold', 3 );

% seed for data stratification
seed = myProcessOptions(op, 'seed', 1);

% Grid search to choose best (xw, yw)
I = strafolds(n, fold, seed );

% Matrix of regression square loss for each fold
CFErr = zeros(fold, length(xwlist), ...
    length(reglist));

for fi=1:fold
    % Make test index
    teI = I(fi, :);
    % Make training index
    trI = ~teI;
    ntr = sum(trI);
%     nte = sum(teI);
    Xtr = X(:, trI);
    Xte = X(:, teI);
    Ztr = Z(:, trI);
    Zte = Z(:, teI);
    
    Str = DistNormal.normalSuffStat(Ztr);
%     Htr = Str'*Str;
    Ste = DistNormal.normalSuffStat(Zte);
%     Hrs = Str'*Ste;
    
    Ste2 = Ste(:)'*Ste(:);
    for xi=1:length(xwlist)
        xw = xwlist(xi)*medx;
        
        Ktr = kerGaussian(Xtr, Xtr, xw);
        Krs = kerGaussian(Xtr, Xte, xw);
        
        for ri=1:length(reglist)
            lambda = reglist(ri);
            A = (Ktr + lambda*eye(ntr)) \ Krs;
            
%             HtrA = Htr*A;
            StrA = (Str*A);
%             HtrA = Str'*(Str*A);
            %sqerr = trace(Hte) + trace(A'*Htr*A) - 2*trace(A'*Hrs);
%             sqerr = Ste(:)'*Ste(:) + A(:)'*HtrA(:) - 2*A(:)'*Hrs(:);
            sqerr = Ste2 + StrA(:)'*StrA(:) - 2*StrA(:)'*Ste(:);
            CFErr(fi, xi,  ri) = sqerr;
            
            fprintf('fold: %d, in_w: %.3g, lamb: %.3g => err: %.3g\n', ...
                fi, xw,  lambda, sqerr);
            
        end
    end
end

CErr = squeeze( mean(CFErr,  1) );
% best param combination
[minerr, ind] = min(CErr(:));
[bxi,  bri] = ind2sub(size(CErr), ind);

CVLog.minerr = minerr;
% best x width
CVLog.bxw = xwlist(bxi);

CVLog.blambda = reglist(bri);

CVLog.medx = medx;


CVLog.reglist = reglist;
CVLog.xwlist = xwlist;

CVLog.seed = seed;
CVLog.fold = fold;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end



