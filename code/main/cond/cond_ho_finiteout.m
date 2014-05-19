function [ C] = cond_ho_finiteout( In, Out, op )
%COND_HO_FINITEOUT Generic repeated hold-out procesure for selecting kernel
%parameters for conditional mean embedding
%   - Learn C_{Out|In}
%   - The conditional mean embedding can take any number of inputs. This
%   simply depends on the Kernel used (kernel on tensor product space).
%   - Use incomplete Cholesky on the kernel matrix. Factorization option
%   can be specified in op (structure).
%   - In is an input Instances objects.
%   - Out must be a matrix where each column represents one instance.
%

assert(nargin>=3, 'op (a structure) is mandatory');
assert(isa(In, 'Instances'));
assert(isnumeric(Out), 'Out must be a matrix');
assert(In.count()==size(Out, 2), 'In and Out must have the same number of instances');

n = size(Out, 2);
% a cell array of Kernel candidates to try.
assert(isfield(op, 'kernel_candidates'), ...
    'Field kernel_candidates is require in op. A cell array of Kernel.');
kernel_candidates = op.kernel_candidates;
assert(all(cellfun(@(k)(isa(k, 'Kernel')) , kernel_candidates) ));

% list of regularization parameters.
reglist = myProcessOptions(op, 'reglist', [1e-2, 1e-0, 10]);

% Number of hold-outs :=h to perform. Train h times on different randomly
% drawn h sets and test on a seperate nonoverlapping test set.
num_ho = myProcessOptions(op, 'num_ho', 5 );
assert(num_ho>=1, 'num_ho (#hold-outs) must be at least 1');

% Training set size. Training and test sets are randomly drawn and non-overlapping.
% Default = 80%
train_size = myProcessOptions(op, 'train_size', floor(0.8*n));

% Test set size. Train and test sizes do not have to add up to n.
% Default = 20%
test_size = myProcessOptions(op, 'test_size',  n-train_size );

% Tolerance for incomplete Cholesky on kernel matrix
chol_tol = myProcessOptions(op, 'chol_tol', 1e-2);

% Maximum rank (column of R) K~R'*R in incomplete Cholesky.
chol_maxrank = myProcessOptions(op, 'chol_maxrank', max(n, 500) );

% seed for data stratification
seed = myProcessOptions(op, 'seed', 1);

% Instances indices
[TRI, TEI] = train_test_indices(num_ho, train_size, test_size, n, seed);

% Matrix of regression square loss. num_ho x reg x candidates
CHErr = inf(num_ho, length(reglist), length(kernel_candidates) );

Z = Out;


% Incomplete Cholesky of the best kernel candidate
% BestKerIChol = [];
% best_err = inf;
% best_ri = 0;
% best_ki = 0;
parfor ki=1:length(kernel_candidates)
    kfunc = kernel_candidates{ki};
    assert(isa(kfunc, 'Kernel'));
    % It is user's responsibility to make sure the kernel can support
    % instances in In.
    % IChol on full kernel matrix
    ichol = IncompChol(In, kfunc, chol_tol, chol_maxrank);
     
    HR = inf(num_ho, length(reglist));
    for hoi=1:num_ho
        % Make test index
        teI = TEI(hoi, :);
        % Make training index
        trI = TRI(hoi, :);
%         ntr = length(trI);
        nte = length(teI);
        Intr = In.get(trI);
        Inte = In.get(teI);
        
        % reduced Cholesky matrix on training set
        Rr = ichol.R(:, trI);
        Krs = kfunc.eval(Intr, Inte); %tr x te
        RrKrs = Rr*Krs;
        Ztr = Z(:, trI);
        Zte = Z(:, teI);
        t1 = Zte(:)'*Zte(:)/nte;
        
        for ri=1:length(reglist)
            lambda = reglist(ri);
            B = (Rr*Rr' + lambda*eye(size(Rr,1))) \ RrKrs;
            Q = (Ztr*Krs - (Ztr*Rr')*B )/lambda; % dz x te
            t2 = Q(:)'*Q(:)/nte;
            t3 = -2*Zte(:)'*Q(:)/nte;
            
            mse = t1+t2+t3;
%             CHErr(hoi, ri, ki) = mse;
            HR(hoi, ri) = mse;
            
            fprintf('hold-out: %d, lamb: %.3g, ker: %s => err: %.3g\n', ...
                hoi, lambda, kfunc.shortSummary(), mse);
        end
    end
    CHErr(:, :, ki) = HR;
%     [minerr, ind] = min(HR(:));
%     [bri, bki] = ind2sub(size(HR), ind);
%     if minerr < best_err
%         best_err = minerr;
%         best_ri = bri;
%         best_ki = bki;
%         BestKerIChol = ichol;
%     end
    
%     CHErr(:, :, ki) = HR;
%     % find minerr for every ki just to avoid storing all Cholesky
%     % factorization results.
%     CErr = shiftdim( mean(CHErr,  1) , 1);
%     % best param combination
%     [minerr, ind] = min(CErr(:));
%     [bri, bki] = ind2sub(size(CErr), ind);
%     if bki==ki
%         BestKerIChol = ichol;
%     end
    
end

CErr = shiftdim( mean(CHErr,  1) , 1);
% best param combination
[minerr, ind] = min(CErr(:));
[bri, bki] = ind2sub(size(CErr), ind);

% At this point, we have bri, bki and BestKerIChol.
C.minerr = minerr;
C.bkernel = kernel_candidates{bki};
% Do Cholesky again. We could have stored but that will disable parfor ?
BestKerIChol = IncompChol(In, C.bkernel, chol_tol, chol_maxrank);
C.bkernel_ichol = BestKerIChol;
C.blambda = reglist(bri);

C.reglist = reglist;
C.kernel_candidates = kernel_candidates;

C.num_ho = num_ho;
C.train_size = train_size;
C.chol_tol = chol_tol;
C.chol_maxrank = chol_maxrank;
C.seed = seed;
C.train_indices = TRI;
C.test_indices = TEI;

end


function [TRI, TEI] = train_test_indices(num_ho, tr, te, n, seed)
% tr = training size, te = test size
% TRI = num_ho x tr (subset of 1:n)
% TEI = num_ho x te = subset of 1:n but disjoint from TRI
assert(tr+te<=n, 'train + test sizes must not exceed n');
% change seed
oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);

TRI = zeros(num_ho, tr);
TEI = zeros(num_ho, te);
for i=1:num_ho
    I = randperm(n);
    TRI(i, :) = I(1:tr);
    TEI(i, :) = I( (tr+1):(tr+te));
end

% change seed back
RandStream.setGlobalStream(oldRs);

end
