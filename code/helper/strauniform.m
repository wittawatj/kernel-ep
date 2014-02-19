function I = strauniform( Y, perclass, seed)
%
% Stratified sampling assuming that the class distribution of Y is uniform.
% That is, randomly select elements in Y so that each class has "perclass"
% points. "perclass" must be positive integer. 
% I is a binary row vector indicating which elements in Y are chosen.
%
%
n = length(Y);

if perclass <= 0
    error('%s: perclass must be a strictly positive integer.', mfilename);
end

oldRs = RandStream.getDefaultStream();
rs = RandStream.create('mt19937ar','seed',seed);
RandStream.setDefaultStream(rs);          


I = false(1,n); % 1 byte per entry for logical matrix
UY = unique(Y);
for ui=1:length(UY)
    y = UY(ui);
    Indy = find(Y==y);
    Indy = Indy(randperm(length(Indy)));
    
    chosenI = Indy(1:perclass); 
    I(chosenI) = true;
    
end
RandStream.setDefaultStream(oldRs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end