function s=funcs_exp4()
    % Return a struct containing functions for exp3
    s=struct();
    s.getAllFileStructs = @getAllFileStructs;
    s.findAllDatasets = @findAllDatasets;
    s.findAllLearners = @findAllLearners;
    s.findMaxTrial = @findMaxTrial;
    s.getDataFileStructs = @getDataFileStructs;
    s.genDNMTStructs = @genDNMTStructs;
    s.printResultTables = @printResultTables;
    s.plotErrorVsNtr = @plotErrorVsNtr;
end

function plotErrorVsNtr(merge)
    % For each data, plot err vs ntr for all learners.
    % DNMT table
    m = merge;
    display(sprintf('max trials: %d ', merge.maxtrial) );
    ntrs = m.ntrs;
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

        %sd_cell = {NMT.log_kl_sd};
        %sI = cellfun(@isempty, sd_cell);
        %sd_cell(sI) = {nan};
        %log_kl_sds = reshape(sd_cell, size(NMT));
        %log_kl_sds = cell2mat(log_kl_sds);
        %NMsd = nanmean(log_kl_sds, 3);
        NMsd = nanstd(log_kl_means, 1, 3);

        figure 
        hold all 
        set(gca, 'fontsize', 20);
        for mi=1:length(m.learners)
            style = Plot.learnerStyle(m.learners{mi});
            errorbar(ntrs', NMmean(:, mi), NMsd(:, mi), style{:});
            %errorbar(repmat(ntrs', 1, length(m.learners)), NMmean, NMsd, ...
            %'linewidth', 2);
        end
        xlabel('Training size');
        ylabel('Log KL error');
        datPretty = Plot.mapDataName(dat);
        title(sprintf('%s', datPretty));
        learnerLegend = cellfun(@Plot.mapLearnerName, m.learners, 'UniformOutput', false);
        legend(learnerLegend{:});
        grid on
        hold off


    end

end

function printResultTables(merge)
    % Generate result tables from the merged result
    % See genDNMTStructs()
    assert(isstruct(merge));
    m = merge;
    replaceFunc = @(l)regexprep(l, 'Learner', '');
    shortLearners = cellfun(replaceFunc, m.learners, 'UniformOutput', false);
    display(sprintf('max trials: %d ', merge.maxtrial) );
    for di=1:length(m.data)
        dat = m.data{di};
        for ni=1:length(m.ntrs)
            strTable = cell(length(m.data), length(m.learners));
            PT = PrintTable();
            PT.HasRowHeader = true;
            PT.HasHeader = true;
            PT.addRow('', shortLearners{:});

            display(sprintf('\n/// printing: %s, ntr=%d ///\n', dat, m.ntrs(ni)))
            % method x trials
            MT = shiftdim(m.table(di, ni, :, :), 2);
            mean_cell = {MT.log_kl_mean};
            mI = cellfun(@isempty, mean_cell);
            mean_cell(mI) = {nan};
            log_kl_means = reshape(mean_cell, size(MT));
            log_kl_means = cell2mat(log_kl_means);
            log_kl_means = nanmean(log_kl_means, 2)';

            sd_cell = {MT.log_kl_sd};
            sI = cellfun(@isempty, sd_cell);
            sd_cell(sI) = {nan};
            log_kl_sds = reshape(sd_cell, size(MT));
            log_kl_sds = cell2mat(log_kl_sds);
            log_kl_sds = nanmean(log_kl_sds, 2)';

            strFunc = @(m, sd)(sprintf('%0.2e (%0.2e)', m, sd));
            row = arrayfun(strFunc, log_kl_means, log_kl_sds, 'UniformOutput', false);
            strTable(di, :) = row;

            ptRow = [m.data{di}, row];
            PT.addRow(ptRow{:});

            matrix2latex(strTable, 1, 'rowLabels', m.data, 'columnLabels', m.learners);
            display('=================================');
            PT.Caption = 'Means of Log KL with SD';
            PT.print();
        end
    end
end

function merge=genDNMTStructs()
    % Generate a 4d struct array table of data x ntr x method x trials 
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
    small = true;
    Data = findAllDatasets(small)
    Methods = findAllLearners(small)
    ntrs = findAllNtr(small)
    maxtrial = findMaxTrial(small)
    T = struct('log_kl_mean', [], 'log_kl_sd', [], 'trN', [], 'teN', [], 'improper_count', []);
    allStructs = getAllFileStructs(small);
    for i=1:length(allStructs)
        fst = allStructs(i);
        %name: 'nvary-ICholMapperLearner-sigmoid_bw_proposal_50000-ntr10000-tri5.mat'
        %learner: 'ICholMapperLearner'
        %data: 'sigmoid_bw_proposal_50000'
        %ntr: '10000'
        %trial: 5
        fpath = Expr.expSavedFile(4, fst.name);
        load(fpath); % this produces a struct s 
        display(sprintf('loaded: %s', fpath));

        %dm = s.dist_mapper;
        % output from the learned operator
        %outDa = s.out_distarray;
        % test messag bundle 
        %teBundle = s.teBundle;
        %trueOutDa = teBundle.getOutBundle();
        %divTester = DivDistMapperTester(dm);
        %divTester.opt('div_function', 'KL');
        %divs = divTester.getDivergence(outDa, trueOutDa);
        divs = s.divs;

        log_kl_mean = nanmean(log(divs));
        log_kl_sd = nanstd(log(divs));
        trN = s.trN;
        teN = s.teN;
        improper_count = sum(isnan(divs));

        di = find(cellfun(@(x)isequal(fst.data, x), Data));
        ni = find(str2double(fst.ntr)==ntrs);
        mi = find(cellfun(@(m)isequal(fst.learner, m), Methods));
        ti = fst.trial;

        entry = struct('log_kl_mean', log_kl_mean, 'log_kl_sd', log_kl_sd, 'trN', trN, ...
            'teN', teN, 'improper_count', improper_count);
        T(di, ni, mi, ti) = entry;

        clear s;
    end
    dest = Expr.expSavedFile(4, 'merge-nvary-exp4.mat');

    merge = struct();
    merge.data = Data;
    merge.learners = Methods;
    merge.maxtrial = maxtrial;
    merge.ntrs = ntrs;
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
        prefix = 'nvary_small';
    else
        prefix = 'nvary';
    end
    expFolder = Expr.expSavedFolder(4);
    files = dir(fullfile(expFolder, [prefix, '-*-*'] ));
    S = struct('name', '', 'learner', '', 'data', '', 'ntr', '', 'trial', '');
    pattern = [prefix, '[-](?<learner>[\w_\d]+)[-](?<data>[\w_\d]+)[-]ntr(?<ntr>\d+)[-]tri(?<trial>\d+)'];
    for i=1:length(files)
        file = files(i);
        Sep = regexp(file.name, pattern , 'names');
        data = Sep.data;
        ntr = Sep.ntr;
        % trial number 
        trial = Sep.trial;
        learner = Sep.learner;

        s = struct();
        s.name = file.name;
        s.learner = learner;
        s.data = data;
        s.ntr = ntr;
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

function C=findAllNtr(small)
    % Find all ntr 
    %
    S = getAllFileStructs(small);
    C = unique(str2double({S.ntr}));
end
