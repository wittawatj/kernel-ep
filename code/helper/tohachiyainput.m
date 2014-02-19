function [ xlabel, xunlabel] = tohachiyainput( X, Y)
%
% Convert inputs used in SMIR paper to Dr.Hachiya's format.
%

UY = unique(Y);
c = length(UY);

for ui=1:c
    uy = UY(ui);
    xlabel(ui).data = X(:, Y==uy);
end
xunlabel = X(:, (length(Y)+1):end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

