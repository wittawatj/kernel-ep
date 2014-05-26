function CVLog = cond_embed_cv2(X, Y, Z, op)
%
% Cross validation for conditional mean embedding for two conditioning
% variables. The embedding operator takes the form C_{z|xy} where X, Y are
% conditioning variables. Assume that Z is generated from p(Z|X,Y)
% - Assume Gaussian kernels for X, Y.
% - Z does not have a kernel. A finite dimensional map (z, z^2) to compute suff stat
% for Gaussian is used.
% - All data matrices are dim x dataset size
%

[d,n] = size(X);

if nargin < 4
    op = [];
end
% list of regularization parameters. Should some how depend on n ??
reglist = myProcessOptions(op, 'reglist', [1e-2, 1e-0, 10]);
% list of Gaussian widths for X. This is to be multiplied with pairwise
% median distance.
list = [1, 2, 4];
xwlist = myProcessOptions(op, 'xwlist', list);
ywlist = myProcessOptions(op, 'ywlist', list);

medx = meddistance(X)^2;
medy = meddistance(Y)^2;

% Number of folds in cross validation
fold = myProcessOptions(op, 'fold', 3 );

% seed for data stratification
seed = myProcessOptions(op, 'seed', 1);

% Grid search to choose best (xw, yw, zw)
I = strafolds(n, fold, seed );

% Matrix of regression square loss for each fold
CFErr = zeros(fold, length(xwlist), ...
    length(ywlist), length(reglist));

for fi=1:fold
    % Make test index
    teI = I(fi, :);
    % Make training index
    trI = ~teI;
    ntr = sum(trI);
%     nte = sum(teI);
    Xtr = X(:, trI);
    Xte = X(:, teI);
    Ytr = Y(:, trI);
    Yte = Y(:, teI);
    Ztr = Z(:, trI);
    Zte = Z(:, teI);
    
    Str = DistNormal.suffStat(Ztr);
%     Htr = Str'*Str;
    Ste = DistNormal.suffStat(Zte);
%     Hrs = Str'*Ste;
    Ste2 = Ste(:)'*Ste(:);
    
    for xi=1:length(xwlist)
        xw = xwlist(xi)*medx;
        Ktr = kerGaussian(Xtr, Xtr, xw);
        Krs = kerGaussian(Xtr, Xte, xw);
        
        for yi=1:length(ywlist)
            yw = ywlist(yi)*medy;
            Ltr = kerGaussian(Ytr, Ytr, yw);
            Lrs = kerGaussian(Ytr, Yte, yw);
            
            KLtr = Ktr.*Ltr;
            KLrs = Krs.*Lrs;
            
            for ri=1:length(reglist)
                lambda = reglist(ri);
                A = (KLtr + lambda*eye(ntr)) \ KLrs;
                StrA = (Str*A);
                % poor performance. Improve later.
                %sqerr = trace(Hte) + trace(A'*Htr*A) - 2*trace(A'*Hrs);
%                 HtrA = Htr*A;
                %   sqerr = Ste2 + A(:)'*HtrA(:) - 2*A(:)'*Hrs(:);
                sqerr = Ste2 + StrA(:)'*StrA(:) - 2*StrA(:)'*Ste(:);
                
                CFErr(fi, xi, yi, ri) = sqerr;
                fprintf('fold: %d, xw: %.3g, yw: %.3g, lamb: %.3g => err: %.3g\n', ...
                    fi, xw, yw, lambda, sqerr);
            end
        end
    end
end

CErr = shiftdim( mean(CFErr,  1) , 1);
% best param combination
[minerr, ind] = min(CErr(:));
[bxi, byi, bri] = ind2sub(size(CErr), ind);

CVLog.minerr = minerr;
% best x width
CVLog.bxw = xwlist(bxi);
CVLog.byw = ywlist(byi);
CVLog.blambda = reglist(bri);

CVLog.medx = medx;
CVLog.medy = medy;

CVLog.reglist = reglist;
CVLog.xwlist = xwlist;
CVLog.ywlist = ywlist;

CVLog.seed = seed;
CVLog.fold = fold;
CVLog.I = I;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
