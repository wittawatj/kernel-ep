function I = straportion( Y, portion, seed, fold)
%
% Stratified sampling.
% Perform subsampling on label vector Y in such a way that the class 
% proportion is maintained.  
% I is a binary row vector indicating which elements in Y are chosen.
%
% 0 < portion < 1
% 
% fold is an optional parameter specifying the minimum number of instances
% for each class needed to retain. This will cause a shift in the class
% balance. The subsample may be over the specified portion. 
% Useful when a cross-validation is to be performed on the stratified
% sample. fold is default to 1. 
%
n = length(Y);

if nargin < 4
    fold = 1;
end
if portion <= 0 || portion >= 1
    error('%s: portion must be between 0 and 1 (exclusive)', mfilename);
end

oldRs = RandStream.getGlobalStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setGlobalStream(rs);


I = false(1,n); % 1 byte per entry for logical matrix
UY = unique(Y);
for ui=1:length(UY)
    y = UY(ui);
    Indy = find(Y==y);
    lindy = length(Indy);
    Indy = Indy(randperm(lindy));
    
    % ceil guarantees that at least 1 instance will be chosen from each
    % class
    to = max(fold, ceil(portion*lindy) );
    
    chosenI = Indy(1:to); 
    I(chosenI) = true;
    
end
RandStream.setGlobalStream(oldRs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end