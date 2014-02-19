base = pwd();
fs = filesep();
folders = {'helper', 'test_idea'};
addpath(pwd);

for fi=1:length(folders)
    fol = folders{fi};
    p = [base , fs, fol];
    fprintf('Add path: %s\n', p);
    addpath(p);
end

% addpath(genpath(fullfile(base, 'real')));
% addpath(genpath(fullfile(base, 'other')));

clear fs folders fol p  fi base expfolders fname i exps
