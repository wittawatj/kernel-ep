function [X, Yc] = smit_input_check( Xi, Yci)
%
% Check input X (dxn) and Y (1xl or 1xn) to be given to SMIT functions. 
% Yc has two possible formats:
% 1. (1xl) containing {1,2,..c}
% 2. (1xn) containing {0,1,2,...c} with 0 denoting 'unlabeled'.
% 
% Format 1. is used in the paper (return format 1).
% Assume the first l columns in X correspond to the labels in Yc. 
% Format 2. is offered for convenience. 
% Format 1. is not allowed to have 0 in Yc.
%

[yr yc] = size(Yci);
[xr xc] = size(Xi);

if yr > 1
    error('Label vector should be a row vector.');
end

if yc < xc 
    % Assume format 1.
    if any(~Yci)
        % contains some zeros
        error('Label vector should not contain 0 in format 1.');
    end
    X = Xi;
    Yc = Yci;
    return;
elseif yc == xc
    % Assume format 2.
    % Make format 1. and return
    I = logical(Yci);
    Yc = Yci(I);
    X = [Xi(:,I), Xi(:,~I)];
    return;
else % yc > xc
    error('Number of labels should not be larger than data points.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

