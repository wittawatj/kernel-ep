% Assume in the environment 
%   Name            Size            Bytes  Class     Attributes
% 
%   N             100x1             11400  cell                
%   VarName1      100x1               800  double              
%   VarName2      100x1               800  double              
%   VarName3      100x1               800  double              
%   VarName4      100x1               800  double              
%   VarName5      100x1               800  double              
%   VarName6      100x1               800  double              
%   VarName7      100x1               800  double              
%   VarName8      100x1               800  double              
%   VarName9      100x1               800  double              
% https://archive.ics.uci.edu/ml/datasets/Fertility
% by importing the data with GUI

X = [VarName1, VarName2, VarName3, VarName4, VarName5, VarName6, VarName7, VarName8, VarName9]';
Io = cellfun(@(c)c=='O', N);
Y = zeros(1, size(X, 2));
Y(Io) = 1.0;
Y(~Io) = 0;