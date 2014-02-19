function I = strafolds(n, fold, seed )
% 
% Return a binary matrix such that each row i indicates the elements used
% in fold i for cross validation. Almost equal proportion of elements is
% maintained across folds.
% 

oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);          


% smallest num larger than n and is divisible by fold
up = ceil(n/fold)*fold;
Ind = repmat( 1:fold , up/fold, 1);
Ind = Ind(:);
Ind = Ind(randperm(up));
Ind = Ind(1:n);

I = false(fold,n); % 1 byte per entry for logical matrix

for i=1:fold
    I(i,:) = Ind==i;
end

RandStream.setGlobalStream(oldRs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

