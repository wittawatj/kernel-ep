function [ s] = globalOptions( )
%GLOBALOPTIONS Struct containing global options for kernel-ep
%   These global options can be overriden when they are used.


s=struct();

% to use multicore or not by default.
s.use_multicore=true;

[p,f,e]=fileparts(which('startup.m'));
% same location as startup.m appended with /tmp
multicoreDir=fullfile(p, 'tmp');
% folder used by multicore package to communicate with other Matlab slaves
s.multicoreDir=multicoreDir;


end

