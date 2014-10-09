function s=funcs_exp5()
    % Return a struct containing functions for exp3
    s=struct();
    s.getAllFileStructs = @getAllFileStructs;
    s.findAllDatasets = @findAllDatasets;
    s.findAllLearners = @findAllLearners;
    s.findMaxTrial = @findMaxTrial;
    s.getDataFileStructs = @getDataFileStructs;
    s.genDNMTStructs = @genDNMTStructs;
    s.printResultTables = @printResultTables;
    s.plotErrorVsNf = @plotErrorVsNf;
end

function plotErrorVsNf(merge)
    % For each data, plot err vs nf (number of random features) for all learners.
    % No Incomplete Cholesky as it does not use random features.
    % DNMT table
    m = merge;
    display(sprintf('max trials: %d ', merge.maxtrial) );
    ntr = m.trN;
    nte = m.teN;
    nfs = m.nfs;
    for di=1:length(m.data)

        dat = m.data{di};
        NMT = shiftdim(m.table(di, :, :, :), 1);

        mean_cell = {NMT.log_kl_mean};
        mI = cellfun(@isempty, mean_cell);
        mean_cell(mI) = {nan};
        log_kl_means = reshape(mean_cell, size(NMT));
        log_kl_means = cell2mat(log_kl_means);
        % average over trials
        NMmean = nanmean(log_kl_means, 3);
        NMsd = nanstd(log_kl_means, 1, 3);

        figure 
        hold all 
        set(gca, 'fontsize', 20);
        for mi=1:length(m.learners)
            style = Plot.learnerStyle(m.learners{mi});
            errorbar(nfs', NMmean(:, mi), NMsd(:, mi), style{:});
            %errorbar(repmat(ntrs', 1, length(m.learners)), NMmean, NMsd, ...
            %'linewidth', 2);
        end
        xlabel('# Random Features');
        ylabel('Log KL error');
        datPretty = Plot.mapDataName(dat);
        title(sprintf('%s. Train/test = %d/%d', datPretty, ntr, nte));
        learnerLegend = cellfun(@Plot.mapLearnerName, m.learners, 'UniformOutput', false);
        legend(learnerLegend{:});
        xlim([min(nfs)-200, max(nfs)+200]);
        grid on
        hold off


    end

end

function merge=genDNMTStructs()
    % Generate a 4d struct array table of data x nf x method x trials 
    %
    %       divs: [1x5000 double]
    %improper_count: 1253
    % learner_class: 'RFGJointEProdLearner'
    %   result_path: [1x136 char]
    %        commit: '8264b091'
    %     timeStamp: [2014 10 4 8 7 40.9490]
    %           trN: 10000
    %           teN: 5000
    %            nf: 7000
    %      trialNum: 2
    %    bundleName: 'sigmoid_fw_proposal_50000'
    small = true;
    Data = findAllDatasets(small)
    Methods = findAllLearners(small)
    nfs = findAllNf(small);
    maxtrial = findMaxTrial(small)
    T = struct('log_kl_mean', [], 'log_kl_sd', [], 'nf', [], 'trN', [], ...
        'teN', [], 'improper_count', []);
    allStructs = getAllFileStructs(small);
    for i=1:length(allStructs)
        fst = allStructs(i);
        fpath = Expr.expSavedFile(5, fst.name);
        load(fpath); % this produces a struct s 
        display(sprintf('loaded: %s', fpath));

        divs = s.divs;

        log_kl_mean = nanmean(log(divs));
        log_kl_sd = nanstd(log(divs));
        nf = s.nf;
        trN = s.trN;
        teN = s.teN;
        improper_count = sum(isnan(divs));

        di = find(cellfun(@(x)isequal(fst.data, x), Data));
        ni = find(str2double(fst.nf)==nfs);
        mi = find(cellfun(@(m)isequal(fst.learner, m), Methods));
        ti = fst.trial;

        entry = struct('log_kl_mean', log_kl_mean, 'log_kl_sd', log_kl_sd, 'nf', nf, ...
            'trN', trN, 'teN', teN, 'improper_count', improper_count);
        T(di, ni, mi, ti) = entry;
        clear s;
    end
    dest = Expr.expSavedFile(5, 'merge-nfvary-exp5.mat');

    merge = struct();
    merge.data = Data;
    merge.learners = Methods;
    merge.maxtrial = maxtrial;
    merge.nfs = nfs;
    merge.trN = trN; %fixed in exp5
    merge.teN = teN; %fixed in exp5
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

function S=getAllFileStructs(small)
    % get all files with information stored in struct 
    % small = true => get the small version
    %
    if nargin < 1
        small =false;
    end

    if small 
        prefix = 'nfvary_small';
    else
        prefix = 'nfvary';
    end
    expFolder = Expr.expSavedFolder(5);
    files = dir(fullfile(expFolder, [prefix, '-*-*'] ));
    S = struct('name', '', 'learner', '', 'data', '', 'nf', '', 'trial', '');
    pattern = [prefix, '[-](?<learner>[\w_\d]+)[-](?<data>[\w_\d]+)[-]nf(?<nf>\d+)[-]tri(?<trial>\d+)'];
    for i=1:length(files)
        file = files(i);
        Sep = regexp(file.name, pattern , 'names');
        data = Sep.data;
        nf = Sep.nf;
        % trial number 
        trial = Sep.trial;
        learner = Sep.learner;

        s = struct();
        s.name = file.name;
        s.learner = learner;
        s.data = data;
        s.nf = nf;
        s.trial = str2double(trial);
        S(i) = s;
    end

end

function m=findMaxTrial(small)
    % return the maximum trial number 
    %
    S = getAllFileStructs(small);
    m = max([S.trial]);
end

function C=findAllDatasets(small)
    % Find the names of all datasets
    %
    S = getAllFileStructs(small);
    C = unique({S.data});
end

function C=findAllLearners(small)
    % Find the names of all learners from the result files 
    %
    S = getAllFileStructs(small);
    C = unique({S.learner});
end

function C=findAllNf(small)
    % Find all nf (number of random features)
    %
    S = getAllFileStructs(small);
    C = unique(str2double({S.nf}));
end
