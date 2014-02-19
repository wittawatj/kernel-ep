function [ nX, nY] = shuffler(X, Y, maxn, seed)
%
% Data shuffler. Randomness depends on the specified seed.
%

% Set the RandStream to use the seed
oldRs = RandStream.getDefaultStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setDefaultStream(rs);          

n = size(X,2);
portion = min(maxn,n)/n;
if portion == 1
    I = randperm(n);
else
    I = straportion( Y, portion , seed);
end

nX = X(:,I);
nY = Y(:,I);

% Set RandStream back to its original one
RandStream.setDefaultStream(oldRs);


%%%%%%%%%%%%%%%%%%%%%
end

