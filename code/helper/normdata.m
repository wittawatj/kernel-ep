function XNorm = normdata( X, Mean, Std )
% 
% Normalize data with the specified Mean and Std vectors.
%
if nargin < 2
    Mean = mean(X,2);
    Std = std(X,0,2);
end

d = size(X,1);
XNorm = bsxfun(@plus, X, -Mean);
denuStd = 1./Std;
% In case of 0 S.D, make the feature 0 
denuStd(isinf(denuStd)) = 0;
DStd = sparse(1:d,1:d, denuStd, d, d);

XNorm = DStd*XNorm;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

