function C = cond_egauss_cv2(X, Y, Z, op)
%
% Cross validation for conditional mean embedding taking 2 incoming messages 
% and outputing a Gaussian distribution. 
% The embedding operator takes the form C_{z|xy} where X, Y are the
% conditioning variables. 
% 
% - X, Y, Z are 1xN DistNormal() array.
% The output mapping for Z represents sufficient statistic for a normal
% distribution, [z, z^2] (finite-dimensional mapping, no parameters).
%


if nargin < 4
    op = [];
end

[~,nx] = size(X);
[~,ny] = size(Y);
[~,nz] = size(Z);
assert(nx==ny);
assert(ny==nz);
assert(isa(X, 'DistNormal'));
assert(isa(Y, 'DistNormal'));
assert(isa(Z, 'DistNormal'));
n = nx;

% list of regularization parameters. Should some how depend on n ??
reglist = myProcessOptions(op, 'reglist', [1e-2, 1e-0, 10]);
% list of Gaussian widths for X. This is to be multiplied with pairwise
% median distance.
list = [1, 2, 4];
xwlist = myProcessOptions(op, 'xwlist', list);
ywlist = myProcessOptions(op, 'ywlist', list);

% normalized median distance
medx = 1;
medy = 1;

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
    
    Str =  dist2suff(Ztr);
%     Htr = Str'*Str;
    Ste = dist2suff(Zte);
%     Hrs = Str'*Ste;
    Ste2 = Ste(:)'*Ste(:);
   
    for xi=1:length(xwlist)
        xw = xwlist(xi)*medx;
     
        Ktr = kerEGaussian(Xtr, Xtr, xw);
        Krs = kerEGaussian(Xtr, Xte, xw);
        
        for yi=1:length(ywlist)
            yw = ywlist(yi)*medy;
            Ltr = kerEGaussian(Ytr, Ytr, yw);
            Lrs = kerEGaussian(Ytr, Yte, yw);
            
            KLtr = Ktr.*Ltr;
            KLrs = Krs.*Lrs;
            
            for ri=1:length(reglist)
                lambda = reglist(ri);
                A = (KLtr + lambda*eye(ntr)) \ KLrs;
                StrA = (Str*A);
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

CErr = shiftdim( mean(CFErr,  1), 1 );
% best param combination
[minerr, ind] = min(CErr(:));
[bxi, byi, bri] = ind2sub(size(CErr), ind);

C.minerr = minerr;
% best x width
C.bxw = xwlist(bxi);
C.byw = ywlist(byi);
C.blambda = reglist(bri);

C.medx = medx;
C.medy = medy;

C.reglist = reglist;
C.xwlist = xwlist;
C.ywlist = ywlist;

C.seed = seed;
C.fold = fold;
C.I = I;

% compute operator and K, L
skx = C.bxw * C.medx; %bxw = best Gaussian width for x
sky = C.byw * C.medy;
K = kerEGaussian(X, X, skx);
L = kerEGaussian(Y, Y, sky);
lamb = C.blambda;

% not a good idea to invert ??
O = inv(K.*L + lamb*eye(n));
C.K = K;
C.L = L;
C.skx = skx;
C.sky = sky;
C.operator = O;
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function S=dist2suff(Z)
% 
assert(isa(Z, 'DistNormal'));
M = [Z.mean];
assert(size(M, 1)==1);

% this will generate an error if Z is multivariate !!!
V = [Z.variance];
% S contains the first two moments
S = [M; V+M.^2];

end



