function C = cond_ggauss_cv2(X, Y, Z, op)
%
% Cross validation for conditional mean embedding taking 2 incoming messages
% and outputing a Gaussian distribution.
% The embedding operator takes the form C_{z|xy} where X, Y are the
% conditioning variables. Use kerGGaussian which takes two parameters
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

% list of regularization parameters. Should somehow depend on n ??
reglist = myProcessOptions(op, 'reglist', [1e-2, 1e-0, 10]);

% list of Gaussian widths (for mean embedding) for X.
list = [1, 2, 4];
xembed_widths = myProcessOptions(op, 'xembed_widths', list);
yembed_widths = myProcessOptions(op, 'yembed_widths', list);

% list of widths for Gaussian kernel on the mean embeddings.
% These are to be multiplied with the median distance heuristic.
xgauss_factors = myProcessOptions(op, 'xgauss_factors', [1]);
ygauss_factors = myProcessOptions(op, 'ygauss_factors', [1]);

% Number of folds in cross validation
fold = myProcessOptions(op, 'fold', 3 );

% seed for data stratification
seed = myProcessOptions(op, 'seed', 1);

% Grid search
I = strafolds(n, fold, seed );

% Matrix of regression square loss for each fold
CFErr = zeros(fold, length(xembed_widths), length(xgauss_factors), ...
    length(yembed_widths), length(ygauss_factors), length(reglist));

% for every embed width, compute the pairwise median distance
% DDx = distance^2 matrix of mean embeddings
[DDx, Med2x] = KGGaussian.compute_meddistances(X, xembed_widths);
assert(length(Med2x)==length(xembed_widths));
[DDy, Med2y] = KGGaussian.compute_meddistances(Y, yembed_widths);
assert(length(Med2y)==length(yembed_widths));

for fi=1:fold
    % Make test index
    teI = I(fi, :);
    % Make training index
    trI = ~teI;
    ntr = sum(trI);
    %     nte = sum(teI);
%     Xtr = X(:, trI);
%     Xte = X(:, teI);
%     Ytr = Y(:, trI);
%     Yte = Y(:, teI);
    Ztr = Z(:, trI);
    Zte = Z(:, teI);
    
    Str =  dist2suff(Ztr);
    %     Htr = Str'*Str;
    Ste = dist2suff(Zte);
    %     Hrs = Str'*Ste;
    Ste2 = Ste(:)'*Ste(:);
    for xei=1:length(xembed_widths)
        sigx2 = xembed_widths(xei);
        %         [ D2] = distGGaussian( Xtr, X2, sigma2)
        for xfi=1:length(xgauss_factors)
            facx = xgauss_factors(xfi);
            gausswx = facx*Med2x(xei);
            % ### performance can be improved by taking subset of matrix
            % DDx{xei}
            Dx = DDx{xei};
            Ktr = exp( -Dx(trI, trI)./(2*gausswx) );
%             Ktr = kerGGaussian(Xtr, Xtr, sigx2, gausswx);
            Krs = exp( -Dx(trI, teI)./(2*gausswx)); 
%             Krs = kerGGaussian(Xtr, Xte, sigx2, gausswx);
            
            for yei=1:length(yembed_widths)
                sigy2 = yembed_widths(yei);
                
                for yfi=1:length(ygauss_factors)
                    facy = ygauss_factors(yfi);
                    gausswy = facy*Med2y(yei);
                    Dy = DDy{yei};
                    Ltr = exp(-Dy(trI, trI)./(2*gausswy));
%                     Ltr = kerGGaussian(Ytr, Ytr, sigy2, gausswy);
                    Lrs = exp(-Dy(trI, teI)./(2*gausswy));
%                     Lrs = kerGGaussian(Ytr, Yte, sigy2, gausswy);
                    
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

                        CFErr(fi, xei, xfi, yei, yfi, ri) = sqerr;
                        fprintf('fold: %d, xew: %.3g, xfac: %.3g, yew: %.3g, yfac: %.3g, lamb: %.3g => err: %.3g\n', ...
                            fi, sigx2, facx, sigy2, facy, lambda, sqerr);
                    end
                    
                end
            end
            
        end
        
    end
    
    
end

CErr = shiftdim( mean(CFErr,  1), 1 );
% best param combination
[minerr, ind] = min(CErr(:));

[bxei, bxfi, byei, byfi, bri] = ind2sub(size(CErr), ind);

C.minerr = minerr;
% best x width
C.bxembed_width = xembed_widths(bxei);
C.bxgauss_fac = xgauss_factors(bxfi);
C.byembed_width = yembed_widths(byei);
C.bygauss_fac = ygauss_factors(byfi);
C.blambda = reglist(bri);

C.medx = Med2x(bxei);
C.medy = Med2y(byei);

C.reglist = reglist;
C.xembed_widths = xembed_widths;
C.xgauss_factors = xgauss_factors;
C.yembed_widths = yembed_widths;
C.ygauss_factors = ygauss_factors;

C.seed = seed;
C.fold = fold;
C.I = I;

% compute operator and K, L
skx = C.bxgauss_fac * C.medx; %bxw = best Gaussian width for x
sky = C.bygauss_fac * C.medy;
% K = kerGGaussian(X, X, C.bxembed_width, skx);
K = exp(-DDx{bxei}./(2*skx));
% L = kerGGaussian(Y, Y, C.byembed_width, sky);
L = exp(-DDy{byei}./(2*sky));
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



