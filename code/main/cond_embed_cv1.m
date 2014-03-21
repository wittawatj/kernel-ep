function CVLog = cond_embed_cv1(X, Z,  op)
%
% Cross validation for conditional mean embedding for one conditioning
% variable. The embedding operator takes the form C_{z|x} where X is the
% conditioning variable.
% - Assume Gaussian kernels.
% - All data matrices are dim x dataset size
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

warning('Why do we have to cross validate on params for Z here ?');
zwlist = myProcessOptions(op, 'zwlist', [1]); % set to [1] for now

medx = meddistance(X);
medz = meddistance(Z);

% Number of folds in cross validation
fold = myProcessOptions(op, 'fold', 3 );

% seed for data stratification
seed = myProcessOptions(op, 'seed', 1);

% Grid search to choose best (xw, yw)
I = strafolds(n, fold, seed );

% Matrix of regression square loss for each fold
CFErr = zeros(fold, length(xwlist), ...
    length(reglist), length(zwlist) );

for fi=1:fold
    % Make test index
    teI = I(fi, :);
    % Make training index
    trI = ~teI;
    ntr = sum(trI);
    nte = sum(teI);
    Xtr = X(:, trI);
    Xte = X(:, teI);
    Ztr = Z(:, trI);
    Zte = Z(:, teI);
    
    for xi=1:length(xwlist)
        xw = xwlist(xi)*medx;
        
        Ktr = kerGaussian(Xtr, Xtr, xw);
        Krs = kerGaussian(Xtr, Xte, xw);
        
        for ri=1:length(reglist)
            lambda = reglist(ri);
            A = (Ktr + lambda*eye(ntr)) \ Krs;
            
            for zi=1:length(zwlist)
                zw = zwlist(zi)*medz;
          
                Htr = kerGaussian(Ztr, Ztr, zw);
                Hrs = kerGaussian(Ztr, Zte, zw);
              
                HtrA = Htr*A;
                sqerr = nte + A(:)'*HtrA(:) - 2*A(:)'*Hrs(:);
                CFErr(fi, xi,  ri, zi) = sqerr;
                
                fprintf('fold: %d, in_w: %.3g, lamb: %.3g, out_w: %.3g => err: %.3g\n', ...
                    fi, xw,  lambda, zw, sqerr);
            end
        end
    end
end

CErr = squeeze( mean(CFErr,  1) );
% best param combination
[minerr, ind] = min(CErr(:));
[bxi,  bri, bzi] = ind2sub(size(CErr), ind);

CVLog.minerr = minerr;
% best x width
CVLog.bxw = xwlist(bxi);
CVLog.bzw = zwlist(bzi);
CVLog.blambda = reglist(bri);

CVLog.medx = medx;

CVLog.medz = medz;

CVLog.reglist = reglist;
CVLog.xwlist = xwlist;

CVLog.zwlist = zwlist;
CVLog.seed = seed;
CVLog.fold = fold;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
