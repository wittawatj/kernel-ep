function [ fname, param] = kerfuncexplode( kerfuncstr )
%
% Break string representation of a kernel function into 2 parts:
% function name, and parameter (assume there is only 1).
%

% @(a,b)(kerLocalScale(a,b,18))
% fname = LocalScale
% param = 18

Sep = regexp(kerfuncstr, ...
    '@\(.*?,.*?\)\((?<fname>[\w\d_]+?)\(.+?,.+?,(?<param>.+?)\)\)', ...
    'names');
fname = regexprep(Sep.fname, 'ker','');
param = Sep.param;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

