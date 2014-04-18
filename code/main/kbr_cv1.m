function C = kbr_cv1(X, Y,  op)
%
% Cross validation for conditional mean embedding for one conditioning
% variable. The embedding operator takes the form C_{y|x} where X is the
% conditioning variable. Assume that X is generated from p(x|y) and Y is
% generated from an arbitrary (but broad) prior. 
% It is important to note that (x,y) ~ p(x|y)s(y) where s(.) is arbitrary 
% but we want C_{y|x} *NOT* C_{x|y}
% 
% - Assume Gaussian kernels.
% - All data matrices are dim x dataset size
% - Assume the feature map for Y is finite-dimensional given by the 
% sufficient statistic for a normal distribution, 
% [y, y^2] (finite-dimensional mapping, no parameters).
%

[d,n] = size(X);
if nargin < 3
    op = [];
end

% Regularization parameters for kernel sum rule part. Should some how depend on n ??
ksr_reglist = myProcessOptions(op, 'ksr_reglist', [1e-4, 1e-2, 1]);

% Regularization parameters for kernel Bayes rule part.
kbr_reglist = myProcessOptions(op, 'kbr_reglist', [1e-4, 1e-2, 1]);

% Mean feature map of the prior of Y. Needed in kernel sum rule part.
% In general, the mean map is infinite-dimensional. However, in this case,
% we can represent the mean map with a vector because we assumed the
% feature map of Y is finite-dimenisional.
Ysuff = DistNormal.normalSuffStat(Y);
y_mean_map = myProcessOptions(op, 'y_mean_map', mean(Ysuff, 2) );
% y_mean_map = myProcessOptions(op, 'y_mean_map', [3, 150]' );

% list of Gaussian widths for X. This is to be multiplied with pairwise
% median distance.
list = [1/3, 1, 3];
xwlist = myProcessOptions(op, 'xwlist', list);
medx = meddistance(X);

% Number of folds in cross validation
fold = myProcessOptions(op, 'fold', 3 );

% seed for data stratification
seed = myProcessOptions(op, 'seed', 1);

I = strafolds(n, fold, seed );

% Matrix of CV loss objective (discrepancy of prior and marginalized estimated joint)
% for each fold
CFErr = zeros(fold, length(xwlist), length(ksr_reglist), ...
    length(kbr_reglist));

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
    
    Yr = DistNormal.normalSuffStat(Ytr);
    Lr = Yr'*Yr;
    Ys = DistNormal.normalSuffStat(Yte);
    % us and vs are weight vectors. Basically sample weights. In generally
    % not necessaily 1/n.
    us = repmat(1/nte, nte, 1);
    vs = repmat(1/nte, nte, 1);
    qa = Ys*vs;
    
    for xi=1:length(xwlist)
        xw = xwlist(xi)*medx;
        
        Kr = kerGaussian(Xtr, Xtr, xw);
        Krs = kerGaussian(Xtr, Xte, xw);
        KrsUs = Krs*us;
        
        for ksri=1:length(ksr_reglist)
            lambda =  ksr_reglist(ksri);
            B = (Lr + lambda*eye(ntr)) \ (Yr'*y_mean_map);
            Dr = diag(B);
            DKu = Dr*KrsUs;
            Er = Dr*Kr;
            Er2 = Er*Er;
             
            for kbri=1:length(kbr_reglist)
                gamma = kbr_reglist(kbri);
                Er2g = Er2 + gamma*eye(ntr);
                % inv() expensive... Need incomplete Cholesky here.
%                 Jr = inv(Er2g);
                cr = Er*( Er2g \ DKu);
                ra = Yr*cr;
                
                sqerr = sum( (qa-ra).^2 );
                CFErr(fi, xi,  ksri, kbri) = sqerr;
            
                fprintf('fold: %d, in_w: %.3g, ksr: %.3g, kbr: %.3g => err: %.3g\n', ...
                    fi, xw,  lambda, gamma, sqerr);     
            end
        end
        
    end
end

CErr = squeeze( mean(CFErr,  1) );

% best param combination
[minerr, ind] = min(CErr(:));
% bksri = best kernel sum rule index (for regularization parameter)
% bkbri = best kernel Bayes rule index
[bxi,  bksri, bkbri] = ind2sub(size(CErr), ind);

C.minerr = minerr;
% best x width
C.bxw = xwlist(bxi);

C.bksr_reg = ksr_reglist(bksri);
C.bkbr_reg = kbr_reglist(bkbri);

C.medx = medx;

C.ksr_reglist = ksr_reglist;
C.kbr_reglist = kbr_reglist;
C.xwlist = xwlist;
C.y_mean_map = y_mean_map;

C.seed = seed;
C.fold = fold;
C.I = I;

% Compute kernel matrices and operator
skx = C.bxw * C.medx; %bxw = best Gaussian width for x
K = kerGaussian(X, X, skx);
YY = DistNormal.normalSuffStat(Y);
L = YY'*YY;
B = (L + C.bksr_reg*eye(n))\(YY'*y_mean_map);
D = diag(B);
DK = D*K;
% operator
O = DK*( (DK*DK + C.bkbr_reg*eye(n))\D );

% store
C.K = K;
C.L = L;
C.Beta = B;
C.operator = O;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end



