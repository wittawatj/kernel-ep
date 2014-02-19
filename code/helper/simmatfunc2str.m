function [ str ] = simmatfunc2str( simmatFunc)
% simmatFunc to string
%

simstr = func2str(simmatFunc);
Sep = regexp(simstr, ...
'@\(.*?)\((?<fname>[\w\d_]+?)\(.+?,(?<param>.+?)\)\)', ...
'names');
fname = regexprep(Sep.fname, 'simmat','');
param = Sep.param;

str = sprintf('%s(%.3g)', fname, str2double(param));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

