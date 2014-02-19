function [ outs ] = pareval( fun, args, options)
%
% Parallel evaluation of the function (fun) with arguments (args) using
% multiple Matlab's on the same machine. Multicore package can also do 
% this. This function is developed
% to eliminate the need to start slave Matlab's in Multicore package in the
% case that parallel computing across machines is not needed.
% 
% - args is a cell array C of cell arrays S 
% - outs is the cell array of outputs obtained from applying (fun) to each
% S. length(outs) == length(args).
% 
% Wittawat Jitkrittum (June 27, 2011)
% 

finish this later if needed.

if nargin < 3
    options = [];
end

% Number of Matlab's to launch
numinstances = 3;

% begin
varsfile = sprintf('%s_%s_input.mat', tempname, mfilename ); 
outfile = sprintf('%s_%s_output.mat', tempname, mfilename ); 

save(varsfile, 'fun', 'args');

% keyboard
% funstr = func2str(fun);
savecmd = sprintf('save(''%s'',''out'')', outfile);
funcmd = sprintf('load(''%s''); out=cell(1,nargout(fun)); [out{:}]=feval(fun, args{:}); %s', varsfile,savecmd);
cmd = sprintf('matlab -nodesktop -nosplash -r "cd %s; addpath(''%s'');%s;exit" ',...
    pwd(), path(), funcmd);

% Execute
[s,w]=system(cmd);

if s ~= 0 
    display(w);
    error('error in executing %s with %s', func2str(fun), pareval);
end 

% Load output
load(outfile);
outs = out;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

