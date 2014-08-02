function s=funcs_exp3()
    % Return a struct containing functions for exp3
    s=struct();
    s.printSummaryTable=@printSummaryTable;
    s.printDistMappers=@printDistMappers;
    s.getImproperOutputCounts=@getImproperOutputCounts;
    s.getResultPaths=@getResultPaths;
    s.printLatexItemize=@printLatexItemize;
    s.getDistMapperSummaries=@getDistMapperSummaries;
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
        C(i)=length(S(i).imp_out);
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


