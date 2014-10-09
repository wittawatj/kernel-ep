classdef Plot
    %PLOT convenience methods for plotting
    
    properties
    end
    
    methods(Static)
        function cellStyles = learnerStyle(learner)
            % Map a learner into a cell array to be use as options for plot()
            % Example cell = {'linewidth', 2', ....}
            %

            allLearners = {'RFGJointEProdLearner', 'RFGSumEProdLearner', ...
                'ICholMapperLearner', 'RFGProductEProdLearner', 'RFGMVMapperLearner'};
            styles = { 
                {'-rx', 'LineWidth', 2 }, ...
                {'-ko', 'LineWidth', 2 }, ...
                {'-m*', 'LineWidth', 2 }, ...
                {'-b<', 'LineWidth', 2 }, ...
                {'-g^', 'LineWidth', 2 }, ...
            
            };
            for i=1:length(allLearners)
                if strcmp(learner, allLearners{i})
                    cellStyles = styles{i};
                    return;
                end
            end
        end

        function name=mapLearnerName(learner)
            % get a pretty name for DistMapperLearner classes
            allLearners = {'RFGJointEProdLearner', 'RFGSumEProdLearner', ...
                'ICholMapperLearner', 'RFGProductEProdLearner', 'RFGMVMapperLearner'};
            names = {'joint embedding', 'sum kernel', 'ichol product kernel', ...
                'product kernel', 'MV kernel'};
            for i=1:length(allLearners)
                if strcmp(learner, allLearners{i})
                    name = names{i};
                    return;
                end
            end
        end

        function name=mapDataName(data)
            % map raw bundle name into pretty name for plotting/printing results
            %
            plainPats = {'sigmoid_fw_.+', 'sigmoid_bw_.+', 'simplegauss_[\w_\d]*?fw', 'simplegauss_[\w_\d]*?bw'};
            plainNames = {'Logistic FW', 'Logistic BW', 'Gaussian FW', 'Gaussian BW'};
            for i=1:length(plainPats)
                if ~isempty(regexp(data, plainPats{i}))

                    name = plainNames{i};
                    return;
                end
            end
            ldsMatch = regexp(data, 'lds_d(?<d>\d+)_to(?<to>[\w])_[\w_\d]+', 'names');
            if ~isempty(ldsMatch)
                d = str2double(ldsMatch.d);
                to = ldsMatch.to;
                name = sprintf('LDS(d=%d) to %s', d, to);
                return;
            end
        end


    end
    
end

