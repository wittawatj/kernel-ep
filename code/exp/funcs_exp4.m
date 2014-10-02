function s=funcs_exp4()
    % Return a struct containing functions for exp3
    s=struct();
    s.getAllFileStructs = @getAllFileStructs;
    s.findAllDatasets = @findAllDatasets;
    s.findAllLearners = @findAllLearners;
    s.findMaxTrial = @findMaxTrial;
    s.getDataFileStructs = @getDataFileStructs;
    s.genDMTStructs = @genDMTStructs;
end

function merge=genDMTStructs()
    % Generate a 2d struct array table of data x method x trials 
    %
    % result in one result file:
    %%s = 
    %  learner_class: 'RFGJointEProdLearner'
    %learner_options: [1x1 Options]
    %    result_path: [1x137 char]
    %    dist_mapper: [1x1 GenericMapper]
    %    learner_log: [1x1 struct]
    %           divs: [1x5000 double]
    %  out_distarray: [1x1 DistArray]
    %         commit: '9832463f'
    %      timeStamp: [2014 9 28 21 21 57.5216]
    %            trN: 8000
    %            teN: 5000
    %       trialNum: 15
    %     bundleName: 'sigmoid_bw_proposal_50000'
    %
    Data = findAllDatasets()
    Methods = findAllLearners()
    maxtrial = findMaxTrial()
    T = struct('log_kl_mean', [], 'log_kl_sd', [], 'trN', [], 'teN', [], 'improper_count', []);
    allStructs = getAllFileStructs();
    for i=1:length(allStructs)
        fst = allStructs(i);
        %name: 'nvary-ICholMapperLearner-sigmoid_bw_proposal_50000-ntr10000-tri5.mat'
        %learner: 'ICholMapperLearner'
        %data: 'sigmoid_bw_proposal_50000'
        %ntr: '10000'
        %trial: 5
        load(Expr.expSavedFile(4, fst.name)); % this produces a struct s 

        dm = s.dist_mapper;
        % output from the learned operator
        outDa = s.out_distarray;
        % test messag bundle 
        teBundle = s.teBundle;
        trueOutDa = teBundle.getOutBundle();

        divTester = DivDistMapperTester(dm);
        divTester.opt('div_function', 'KL');
        divs = divTester.getDivergence(outDa, trueOutDa);

        log_kl_mean = nanmean(log(divs));
        log_kl_sd = nanstd(log(divs));
        trN = s.trN;
        teN = s.teN;
        improper_count = sum(isnan(divs));

        di = find(cellfun(@(x)isequal(fst.data, x), Data));
        mi = find(cellfun(@(m)isequal(fst.learner, m), Methods));
        ti = fst.trial;

        entry = struct('log_kl_mean', log_kl_mean, 'log_kl_sd', log_kl_sd, 'trN', trN, ...
            'teN', teN, 'improper_count', improper_count);
        T(di, mi, ti) = entry;

        clear s;
    end
    dest = Expr.expSavedFile(4, 'merge-nvary-exp4.mat');

    merge = struct();
    merge.data = Data;
    merge.learners = Methods;
    merge.maxtrial = maxtrial;
    merge.table = T;
    merge.description = 'table is of size data x learners x maxtrial.';
    save(dest, 'merge' );

    
end

function DS=getDataFileStructs(dataset)
    % Get all file structs matching the specified dataset 
    %
    S = getAllFileStructs();
    data = {S.data};
    % bool array
    I = cellfun( @(x)isequal(x, dataset), data);
    DS = S(I);
end

function S=getAllFileStructs()
    % get all files with information stored in struct 
    %

    expFolder = Expr.expSavedFolder(4);
    files = dir(fullfile(expFolder, 'nvary-*-*' ));
    S = struct('name', '', 'learner', '', 'data', '', 'ntr', '', 'trial', '');
    for i=1:length(files)
        file = files(i);
        Sep = regexp(file.name, 'nvary-(?<learner>[\w_\d]+)[-](?<data>[\w_\d]+)[-]ntr(?<ntr>\d+)[-]tri(?<trial>\d+)', 'names');
        learner = Sep.learner;
        data = Sep.data;
        ntr = Sep.ntr;
        % trial number 
        trial = Sep.trial;

        s = struct();
        s.name = file.name;
        s.learner = learner;
        s.data = data;
        s.ntr = ntr;
        s.trial = str2double(trial);
        S(i) = s;
    end

end

function m=findMaxTrial()
    % return the maximum trial number 
    %
    S = getAllFileStructs();
    m = max([S.trial]);
end

function C=findAllDatasets()
    % Find the names of all datasets
    %
    S = getAllFileStructs();
    C = unique({S.data});
end

function C=findAllLearners()
    % Find the names of all learners from the result files 
    %
    S = getAllFileStructs();
    C = unique({S.learner});
end

