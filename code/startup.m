base = pwd();
fs = filesep();
folders = {'helper', 'test_idea', 'main', ...
    'plot', ...
     'main/cond', 'test_idea/clutter', 'script'};
addpath(pwd);

for fi=1:length(folders)
    fol = folders{fi};
    p = [base , fs, fol];
    fprintf('Add path: %s\n', p);
    addpath(p);
end

% folders to be added by genpath
gfolders = {'3rdparty/xunit', 'main/tool'};
for i=1:length(gfolders)
    fol = gfolders{i};
    p = [base , fs, fol];
    fprintf('Add gen path: %s\n', p);
    addpath(genpath(p));
end

% addpath(genpath(fullfile(base, 'real')));
% addpath(genpath(fullfile(base, 'other')));

clear fs gfolders folders fol p  fi base expfolders fname i exps
