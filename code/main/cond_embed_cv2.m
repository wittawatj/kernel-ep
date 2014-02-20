function CVLog = cond_embed_cv2(X, Y, Z, op)
%
% Cross validation for conditional mean embedding for two conditioning
% variables. The embedding operator takes the form C_{z|xy} where X, Y are
% conditioning variables.
% - Assume Gaussian kernels.
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
zwlist = myProcessOptions(op, 'zwlist', list);

medx = meddistance(X);
medy = meddistance(Y);
medz = meddistance(Z);

% Number of folds in cross validation
fold = myProcessOptions(op, 'fold', 3 );

% seed for data stratification
seed = myProcessOptions(op, 'seed', 1);


% Grid search to choose best (xw, yw, zw)
I = strafolds(n, fold, seed );

% Matrix of regression square loss for each fold 
CFErr = zeros(fold, length(xwlist), ...
    length(ywlist), length(reglist), length(zwlist) );

for fi=1:fold
    % Make test index
    teI = I(fi, :);
    % Make training index
    trI = ~teI;
    ntr = sum(trI);
    nte = sum(teI);
    Xtr = X(:, trI);
    Xte = X(:, teI);
    Ytr = Y(:, trI);
    Yte = Y(:, teI);
    Ztr = Z(:, trI);
    Zte = Z(:, teI);
    
    
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
                
                for zi=1:length(zwlist)
                    zw = zwlist(zi)*medz;
                    Hte = kerGaussian(Zte, Zte, zw);
                    Htr = kerGaussian(Ztr, Ztr, zw);
                    Hrs = kerGaussian(Ztr, Zte, zw);
                    % poor performance. Improve later.
                    sqerr = trace(Hte) + trace(A'*Htr*A) - 2*trace(A'*Hrs);
                    CFErr(fi, xi, yi, ri, zi) = sqerr;
                    
                    fprintf('fold: %d, xw: %.3g, yw: %.3g, lamb: %.3g, zw: %.3g => err: %.3g\n', ...
                        fi, xw, yw, lambda, zw, sqerr);
                end
            end
        end
    end
end

CErr = squeeze( mean(CFErr,  1) );
% best param combination
[minerr, ind] = min(CErr(:));
[bxi, byi, bri, bzi] = ind2sub(size(CErr), ind);

CVLog.minerr = minerr;
% best x width
CVLog.bxw = xwlist(bxi);
CVLog.byw = ywlist(byi);
CVLog.bzw = zwlist(bzi);
CVLog.blambda = reglist(bri);

CVLog.medx = medx;
CVLog.medy = medy;
CVLog.medz = medz;

CVLog.reglist = reglist;
CVLog.xwlist = xwlist;
CVLog.ywlist = ywlist;
CVLog.zwlist = zwlist;
CVLog.seed = seed;
CVLog.fold = fold;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end