function [ X, Y] = tosmirinput( xlabel, xunlabel)
%
% Convert inputs in Dr.Hachiya's format to SMIR paper's format.
%
c = length(xlabel);
X = [xlabel.data , xunlabel];
Cn = cellfun(@(x)(size(x,2)), {xlabel.data});
Cy = arrayfun(@(cn, cl)(repmat(cl, 1, cn)), Cn, 1:c,'UniformOutput',false);
Y = [Cy{:}];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

