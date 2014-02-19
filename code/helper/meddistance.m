function dist = meddistance( X )
%
% Find the median distance from all pair-wise
% distance of instances in X. Euclidean distance is used.
% X is a matrix with each column representing one instance.
%
if exist('pdist', 'file')
    dist = median(pdist(X'));
else
    dist = median(nonzeros(tril(ipdm(X'),-1)));
end

% 



end

