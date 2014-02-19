function I = strapart( Y, fold, seed)
%
% Stratified partition.
% Partition Y into 'fold' folds while retaining the proportion of class
% labels in Y. Y must be a row vector.
% I is a foldxn binary fold indicator matrix.
%

oldRs = RandStream.getDefaultStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setDefaultStream(rs);          

n = length(Y);
I = false(fold,n); % 1 byte per entry for logical matrix
UY = unique(Y);
for ui=1:length(UY)
    y = UY(ui);
    Indy = find(Y==y);
    lindy = length(Indy);
    IIy = repmat(1:fold, ceil(lindy/fold),1)';
    IIy = IIy(:);    
    IIy = IIy(1:lindy);
    IIy = IIy(randperm(lindy));
    
    for fi=1:fold
        I(fi, Indy(IIy==fi)) = true;
    end
    
end


RandStream.setDefaultStream(oldRs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end