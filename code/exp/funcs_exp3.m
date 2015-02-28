function s=funcs_exp3()
    % Return a struct containing functions for exp3
    s=struct();
    s.printSummaryTable=@printSummaryTable;
    s.printDistMappers=@printDistMappers;
    s.getImproperOutputCounts=@getImproperOutputCounts;
    s.getResultPaths=@getResultPaths;
    s.printLatexItemize=@printLatexItemize;
    s.getDistMapperSummaries=@getDistMapperSummaries;
    s.getSigmoidFactorOperator=@getSigmoidFactorOperator;
    s.genSigmoidFactorOperator=@genSigmoidFactorOperator;
end

function T=printSummaryTable(S)
    % S is a struct array obtained from the result saved by exp3_
    assert(isstruct(S));
    MLogDivs=nanmean(log(vertcat(S.divs)), 2);
    SdLogDivs=nanstd(log(vertcat(S.divs)), [], 2);
    impCounts=getImproperOutputCounts(S);

    rowLabels={S.learner_class};
    columnLabels={'mean log KL', 's.d. log KL', 'improper count'};
    T=[MLogDivs, SdLogDivs, impCounts];

    % total test msgs
    display(sprintf('\\textbf{Total test messages:} %d \\\\', length(S(1).out_distarray) ));
    matrix2latex(T, 1, 'rowLabels', rowLabels, 'columnLabels', columnLabels);

    % result file paths
    P=getResultPaths(S);
    display(' ');
    display(sprintf('\\textbf{Result paths:} '));
    printLatexItemize(P, '\small');

    % selected DistMapper's
    display(' ');
    display(sprintf('\\textbf{Selected parameters:} '));
    mapperStrs=getDistMapperSummaries(S);
    printLatexItemize(mapperStrs, '\footnotesize');
end

function printLatexItemize(C, textFormat)
    % C is a cell array of string 
    % textFormat can be e.g., \small
    if nargin<2
        textFormat='';
    end

    s='';
    newline=char(10);
    s=[s, '\begin{itemize}', newline];
    for i=1:length(C)
        s=[s,  '\item{ ', textFormat, '\verb|', C{i}, '| } ', newline];
    end
    s=[s, '\end{itemize}', newline ];
    display(sprintf('%s', s));
end

function C=getDistMapperSummaries(S)
    % Return a cell array of shortSummary()'s of all DistMapper's
    %
    l=length(S);
    C=cell(l, 1);
    for i=1:l
        C{i}=S(i).dist_mapper.shortSummary();
    end
end

function P=getResultPaths(S)
    % Return a column cell array of result paths, one for each DistMapperLearner
    %
    l=length(S);
    P=cell(l, 1);
    for i=1:l
        [p,f,e]=fileparts(S(i).result_path);
        [folder, f1, e1]=fileparts(p);
        P{i}=fullfile(f1, [f, e]);
    end

end

function C=getImproperOutputCounts(S)
    % return a column Vector
    C=zeros(length(S), 1);
    for i=1:length(S)
        if isfield(S(i), 'imp_out')
            C(i)=length(S(i).imp_out);
        else 
            C(i) = nan;
        end
    end
end

function printDistMappers(S)
    % S is a struct array obtained from the result saved by exp3_
    assert(isstruct(S));
    for i=1:length(S)
        display(sprintf(S(i).dist_mapper.shortSummary() ));
        display(' ');
    end
end


function fo=getSigmoidFactorOperator(fwFile, bwFile, summary)
    % Construct a FactorOperator for sigmoid problem. 
    % This requires loading from at least two files in exp3 results as 
    % we need one DistMapper for forward and another for backward direction.
    %
    % fwFile = file name in exp3 folder 
    % bwFile = 
    %
    
    fwPath=Expr.expSavedFile(3, fwFile);
    % expect a struct s
    load(fwPath);
    fwResult=s;
    assert(isa(fwResult.out_distarray(1), 'DistBeta'));

    bwPath=Expr.expSavedFile(3, bwFile);
    load(bwPath);
    bwResult=s;
    assert(isa(bwResult.out_distarray(1), 'DistNormal'));
    %  learner_class: 'RFGMVMapperLearner'
    %learner_options: [1x1 Options]
    %    result_path: '/nfs/nhome/live/wittawat/SHARE/gatsby/research/code/saved/exp/exp3/RFGMVMapperLearner_nicolas_sigmoid_fw_25000.mat'
    %    dist_mapper: [1x1 GenericMapper]
    %    learner_log: [1x1 struct]
    %     div_tester: [1x1 DivDistMapperTester]
    %        helling: [1x5000 double]
    %  out_distarray: [1x1 DistArray]
    %     imp_tester: [1x1 ImproperDistMapperTester]
    %        imp_out: [1x2358 DistBeta]
    %         commit: 'b1535cc4'
    %      timeStamp: [2014 7 27 17 30 47.8046]

    distMapper1=fwResult.dist_mapper;
    distMapper2=bwResult.dist_mapper;
    fo=DefaultFactorOperator({distMapper1, distMapper2}, summary);
end


function genSigmoidFactorOperator()
    % convenient method to generate a sigmoid FactorOperator from exp3 results.
    fwFile='RFGMVMapperLearner_nicolas_sigmoid_fw_25000.mat';
    bwFile='RFGMVMapperLearner_nicolas_sigmoid_bw_25000.mat';

    foName='RFGMVMapperLearner_nicolas_sigmoid_25000';

    summary=['sigmoid factor. p(x1|x2) where x1 is Beta and '...
        'x2 is Normal. RFGMVMapperLearner_nicolas_sigmoid_25000'];
    fo=getSigmoidFactorOperator(fwFile, bwFile, summary);
    serializer=FactorOpSerializer();
    % save operator 
    serializer.saveFactorOperator(fo, foName);

    % serialize operator for C#
    serializer.serializeFactorOperator(fo, foName);

end


