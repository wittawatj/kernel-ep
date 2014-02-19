function kerFuncs = kernelCandidates( ker, params)
%
% Create a cell array of kernel functions by applying each parameter
% in params to the third argument of ker. 
% For example, if ker = @kerGaussian, and params = [1 2],
% then return a cell array containing 2 Gaussian kernel functions
% kerFuncs = {@(a,b)(kerGaussian(a,b,1), @(a,b)(kerGaussian(a,b,2)}
%
% - params must be a vector
% - length(kerFuncs) = length(params)
%

% If use this, func2str does not return useful information.
% kerFuncs = arrayfun( @(pa)(@(a,b)(ker(a,b,pa))), params,...
%     'UniformOutput', false);

kerFuncs = cell(1, length(params));
for i=1:length(kerFuncs)
    p = params(i);
    funcstr = sprintf('@(a,b)(%s(a,b,%.10g))', func2str(ker), p);
    kerFuncs{i} = eval(funcstr);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


