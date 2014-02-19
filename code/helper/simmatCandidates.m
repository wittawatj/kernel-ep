function [ simmatFuncs ] = simmatCandidates( sim, params )
%
% Create a cell array of simmat functions by applying each parameter
% in params to the second argument of sim. 
% For example, if sim = @simmatGaussian, and params = [1 2],
% then return a cell array containing 2 Gaussian affinity functions
% simmatFuncs = {@(X)(simmatGaussian(X,1), @(X)(simmatGaussian(X,2)}
%
% - params must be a vector
% - length(simmatFuncs) = length(params)
%

simmatFuncs = cell(1, length(params));
for i=1:length(simmatFuncs)
    p = params(i);
    funcstr = sprintf('@(X)(%s(X,%.10g))', func2str(sim), p);
    simmatFuncs{i} = eval(funcstr);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

