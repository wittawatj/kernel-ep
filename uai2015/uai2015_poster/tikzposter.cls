%%
%% This is file `tikzposter.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% tikzposter.dtx  (with options: `tikzposter.cls')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2014 by Pascal Richter, Elena Botoeva, Richard Barnard, and Dirk Surmann
%% 
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either
%% version 2.0 of this license or (at your option) any later
%% version. The latest version of this license is in:
%% 
%% http://www.latex-project.org/lppl.txt
%% 
%% and version 2.0 or later is part of all distributions of
%% LaTeX version 2013/12/01 or later.
%% 








\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tikzposter}[2014/01/15 v2.0 LaTeX document class for Posters]

 % --------------------------------------- %
 % Loading Packages
\usepackage{xkeyval}
\usepackage{calc}
\usepackage{ifthen}
\usepackage{ae}
\usepackage{xstring}
\usepackage{etoolbox}
\usepackage{tikz}
    \usetikzlibrary{shapes,decorations,shadows,backgrounds,calc,fadings,fit}
    \usetikzlibrary{decorations.pathmorphing}
    \usepgflibrary{arrows}

 % --------------------------------------- %
 % Paper parameter
\newif\ifTP@hugefontsize
\newdimen\TP@innermargin
\newdimen\TP@visibletextwidth
\newdimen\TP@visibletextheight

 % Title parameter
\newdimen\TP@titlewidth
\newdimen\titlewidth
\newdimen\TP@titlelinewidth
\newdimen\titlelinewidth
\def\titleroundedcorners{}
\newdimen\titleinnersep
\newdimen\TP@titleinnersep
\newdimen\TP@titletotopverticalspace
\newdimen\titletotopverticalspace
\newdimen\TP@titletoblockverticalspace
\newdimen\TP@titleheight
\newdimen\titleheight
\newdimen\titlegraphicheight
\newdimen\titleposleft
\newdimen\titleposright
\newdimen\titlepostop
\newdimen\titleposbottom
\newdimen\TP@titlegraphictotitledistance
\def\TP@titletextscale{1}

 % Maketitle parameter
\def\@title{~}
\def\title#1{\gdef\@title{\scalebox{\TP@titletextscale}{#1}}}
\def\@author{~}
\def\@institute{~}
\def\@titlegraphic{~}

 % Affection parameter
\newif\ifTP@showlatexaffection

 % Column parameter
\newif\ifTP@columnEnvironment
\TP@columnEnvironmentfalse
\newif\ifTP@subcolumnEnvironment
\TP@subcolumnEnvironmentfalse
\newdimen\TP@colspace
\newdimen\TP@coltop
\newdimen\TP@colbottom
\newdimen\TP@colcenter
\newdimen\colwidth

 % Subcolumn parameter
\newdimen\TP@subcolspace
\newdimen\TP@subcoltop
\newdimen\TP@subcolbottom
\newdimen\TP@subcolcenter
\newdimen\subcolwidth

 % Block parameter
\newdimen\TP@blockverticalspace
\newdimen\TP@blockcenter
\newdimen\TP@blocktitleinnersep
\newdimen\blocktitleinnersep
\newdimen\TP@blockbodyinnersep
\newdimen\blockbodyinnersep
\newbox\TP@blocktitlebox
\newbox\TP@blockbodybox
\def\TP@blocktitleAlignment{}
\newif\ifBlockHasTitle
\newdimen\TP@blocktitlewidth
\newdimen\TP@blockbodywidth
\newdimen\TP@blockbodyheight
\newdimen\TP@blocktitleheight
\newdimen\TP@blocktop
\newdimen\TP@blocktitleoffsetx
\newdimen\TP@blocktitleoffsety
\newdimen\TP@blockbodyoffsetx
\newdimen\TP@blockbodyoffsety
\newdimen\TP@blockbodyverticalshift
\def\blockroundedcorners{}
\newdimen\blocklinewidth

 % Innerblock parameter
\newdimen\TP@innerblockcenter
\newdimen\TP@innerblocktitleinnersep
\newdimen\innerblocktitleinnersep
\newdimen\TP@innerblockbodyinnersep
\newdimen\innerblockbodyinnersep
\newbox\TP@innerblocktitlebox
\newbox\TP@innerblockbodybox
\def\TP@innerblocktitleAlignment{}
\newif\ifInnerblockHasTitle
\newdimen\TP@innerblocktitlewidth
\newdimen\TP@innerblockbodywidth
\newdimen\TP@innerblockbodyheight
\newdimen\TP@innerblocktitleheight
\newdimen\TP@innerblocktitleoffsetx
\newdimen\TP@innerblocktitleoffsety
\newdimen\TP@innerblockbodyoffsetx
\newdimen\TP@innerblockbodyoffsety
\newdimen\TP@innerblockbodyverticalshift
\def\innerblockroundedcorners{}
\newdimen\innerblocklinewidth

 % Coloredbox parameter
\newbox\TP@coloredbox
\newdimen\TP@coloredboxwidth
\def\TP@coloredboxroundedcorners{}
\newdimen\TP@coloredboxlinewidth
\newdimen\TP@coloredboxinnersep
\newdimen\TP@coloredboxheight

 % Note parameter
\newdimen\TP@noteinnersep
\newdimen\noteinnersep
\newdimen\TP@notetargetoffsetx
\newdimen\TP@notetargetoffsety
\def\TP@noteangle{}
\newdimen\TP@noteradius
\newdimen\TP@notewidth
\newdimen\TP@noteheight
\newif\ifNoteHasConnection
\newbox\TP@notebox
\def\noterotate{}
\def\noteroundedcorners{}
\newdimen\notelinewidth

 % --------------------------------------- %
 % Declaring options
\DeclareOptionX{12pt}{\TP@hugefontsizefalse \PassOptionsToClass{\CurrentOption}{extarticle}}
\DeclareOptionX{14pt}{\TP@hugefontsizefalse \PassOptionsToClass{\CurrentOption}{extarticle}}
\DeclareOptionX{17pt}{\TP@hugefontsizefalse \PassOptionsToClass{\CurrentOption}{extarticle}}
\DeclareOptionX{20pt}{\TP@hugefontsizefalse \PassOptionsToClass{\CurrentOption}{extarticle}}
\DeclareOptionX{25pt}{\TP@hugefontsizetrue}
\DeclareOptionX{fleqn}{\PassOptionsToClass{\CurrentOption}{extarticle}}
\DeclareOptionX{leqno} {\PassOptionsToClass{\CurrentOption}{extarticle}}
\DeclareOptionX{a0paper}{\PassOptionsToPackage{\CurrentOption}{geometry}}
\DeclareOptionX{a1paper}{\PassOptionsToPackage{\CurrentOption}{geometry}}
\DeclareOptionX{a2paper}{\PassOptionsToPackage{\CurrentOption}{geometry}}
\DeclareOptionX{landscape}{\PassOptionsToPackage{\CurrentOption}{geometry}}
\DeclareOptionX{portrait}{\PassOptionsToPackage{\CurrentOption}{geometry}}
\DeclareOptionX{margin}[20mm]{\PassOptionsToPackage{margin=#1}{geometry}}
\DeclareOptionX{innermargin}{\TP@innermargin=#1}
\DeclareOptionX{colspace}{\TP@colspace=#1}
\DeclareOptionX{subcolspace}{\TP@subcolspace=#1}
\DeclareOptionX{blockverticalspace}{\TP@blockverticalspace=#1}
\DeclareOptionX*{{\PackageWarning{tikzposter}{Unknown Option \CurrentOption.}}}

 % Executing options
\ExecuteOptionsX{
    25pt,
    a0paper,
    portrait,
    margin=0mm,
    innermargin=15mm,
    colspace=15mm,
    subcolspace=8mm,
    blockverticalspace=15mm
}
\ProcessOptionsX \relax
\LoadClass{extarticle}
\RequirePackage{geometry}
\ifTP@hugefontsize
    \input{a0size.sty}
    \renewcommand{\tiny}{\fontsize{12}{14}\selectfont}
    \renewcommand{\scriptsize}{\fontsize{14.4}{18}\selectfont}
    \renewcommand{\footnotesize}{\fontsize{17.28}{22}\selectfont}
    \renewcommand{\small}{\fontsize{20.74}{25}\selectfont}
    \renewcommand{\normalsize}{\fontsize{24.88}{30}\selectfont}
    \renewcommand{\large}{\fontsize{29.86}{37}\selectfont}
    \renewcommand{\Large}{\fontsize{35.83}{45}\selectfont}
    \renewcommand{\LARGE}{\fontsize{43}{54}\selectfont}
    \renewcommand{\huge}{\fontsize{51.6}{64}\selectfont}
    \renewcommand{\Huge}{\fontsize{61.92}{77}\selectfont}
\fi
\normalsize
\pagestyle{empty}
\setlength{\TP@visibletextwidth}{\textwidth-2\TP@innermargin}
\setlength{\TP@visibletextheight}{\textheight-2\TP@innermargin}
\pgfdeclarelayer{backgroundlayer}
\pgfdeclarelayer{notelayer}
\pgfsetlayers{backgroundlayer,main,notelayer}

 % --------------------------------------- %
 % Background styles
\gdef\definebackgroundstyle#1#2{\expandafter\gdef\csname TP@backgroundstyle#1\endcsname{#2}}

\gdef\usebackgroundstyle#1{
    \ifcsname TP@backgroundstyle#1\endcsname
         \gdef\TP@backgroundstyle{\csname TP@backgroundstyle#1\endcsname} % save macro
    \else
        \PackageWarning{tikzposter}{Unknown backgroundstyle `#1'.}
    \fi
}

 % --------------------------------------- %
 % Title styles
\gdef\definetitlestyle#1#2#3{
    \expandafter\gdef\csname TP@titlestyle#1Defaultvalues\endcsname{\gpresetkeys{title}{#2}{}}
    \expandafter\gdef\csname TP@titlestyle#1\endcsname{#3}
}

\define@key{title}{width}{\TP@titlewidth=#1 \titlewidth=#1}
\define@key{title}{roundedcorners}{\gdef\titleroundedcorners{#1}}
\define@key{title}{linewidth}{\TP@titlelinewidth=#1 \titlelinewidth=#1}
\define@key{title}{innersep}{\TP@titleinnersep=#1 \titleinnersep=#1}
\define@key{title}{titletotopverticalspace}{\TP@titletotopverticalspace=#1 \titletotopverticalspace=#1}
\define@key{title}{titletoblockverticalspace}{\TP@titletoblockverticalspace=#1}
\define@key{title}{titlegraphictotitledistance}{\TP@titlegraphictotitledistance=#1}
\define@key{title}{titletextscale}{\TP@titletextscale=#1}
\newcommand\usetitlestyle[2][]{
    \ifcsname TP@titlestyle#2\endcsname
         \csname TP@titlestyle#2Defaultvalues\endcsname\gpresetkeys{title}{#1}{} % call macro
         \gdef\TP@titlestyle{\csname TP@titlestyle#2\endcsname} % save macro
    \else
        \PackageWarning{tikzposter}{Unknown titlestyle `#2'.}
    \fi
}

 % --------------------------------------- %
 % Title
\gdef\TP@maketitle{
    \centering
    \vbox{
    \@titlegraphic
        \\[\TP@titlegraphictotitledistance]
        \centering
        \color{titlefgcolor}
        {\bfseries \Huge \sc \@title \par}
        \vspace*{1em}
        {\huge \@author \par}
        \vspace*{1em}
        {\LARGE \@institute}
    }
}

\gdef\institute#1{\gdef\@institute{#1}}
\gdef\titlegraphic#1{\gdef\@titlegraphic{#1}}

\newenvironment{settitle}{
    \makeatletter
    \renewcommand{\TP@maketitle}
}{
    \makeatother
}

\renewcommand\maketitle[1][]{  % #1 keys
    \normalsize
    \setkeys{title}{#1}
    % Title dummy to get title height
    \node[transparent,inner sep=\TP@titleinnersep, line width=\TP@titlelinewidth, anchor=north, minimum width=\TP@visibletextwidth-2\TP@titleinnersep]
        (TP@title) at ($(0, 0.5\textheight-\TP@titletotopverticalspace)$) {\parbox{\TP@titlewidth-2\TP@titleinnersep}{\TP@maketitle}};
    \draw let \p1 = ($(TP@title.north)-(TP@title.south)$) in node {
        \setlength{\TP@titleheight}{\y1}
        \setlength{\titleheight}{\y1}
        \global\TP@titleheight=\TP@titleheight
        \global\titleheight=\titleheight
    };

    % Compute title position
    \setlength{\titleposleft}{-0.5\titlewidth}
    \setlength{\titleposright}{\titleposleft+\titlewidth}
    \setlength{\titlepostop}{0.5\textheight-\TP@titletotopverticalspace}
    \setlength{\titleposbottom}{\titlepostop-\titleheight}

    % Title style (background)
    \TP@titlestyle

    % Title node
    \node[inner sep=\TP@titleinnersep, line width=\TP@titlelinewidth, anchor=north, minimum width=\TP@visibletextwidth-2\TP@titleinnersep]
        at (0,0.5\textheight-\TP@titletotopverticalspace) {\parbox{\TP@titlewidth-2\TP@titleinnersep}{\TP@maketitle}};

    % Settings for blocks
    \normalsize
    \setlength{\TP@blocktop}{\titleposbottom-\TP@titletoblockverticalspace}
}

 % --------------------------------------- %
 % Block styles
\gdef\defineblockstyle#1#2#3{
    \expandafter\gdef\csname TP@blockstyle#1Defaultvalues\endcsname{\gpresetkeys{block}{#2}{}}
    \expandafter\gdef\csname TP@blockstyle#1\endcsname{#3}
}

\define@key{block}{titlewidthscale}{\setlength{\TP@blocktitlewidth}{#1\TP@blocktitlewidth}}
\define@key{block}{bodywidthscale}{\setlength{\TP@blockbodywidth}{#1\TP@blockbodywidth}}
\define@boolkey{block}{titleleft}[true]{\ifKV@block@titleleft \gdef\TP@blocktitleAlignment{\raggedright} \fi}
\define@boolkey{block}{titlecenter}[true]{\ifKV@block@titlecenter \gdef\TP@blocktitleAlignment{\centering} \fi}
\define@boolkey{block}{titleright}[true]{\ifKV@block@titleright \gdef\TP@blocktitleAlignment{\raggedleft} \fi}
\define@key{block}{titleoffsetx}{\TP@blocktitleoffsetx=#1}
\define@key{block}{titleoffsety}{\TP@blocktitleoffsety=#1}
\define@key{block}{bodyoffsetx}{\TP@blockbodyoffsetx=#1}
\define@key{block}{bodyoffsety}{\TP@blockbodyoffsety=#1}
\define@key{block}{bodyverticalshift}{\TP@blockbodyverticalshift=#1}
\define@key{block}{roundedcorners}{\gdef\blockroundedcorners{#1}}
\define@key{block}{linewidth}{\blocklinewidth=#1}
\define@key{block}{titleinnersep}{\TP@blocktitleinnersep=#1 \blocktitleinnersep=#1}
\define@key{block}{bodyinnersep}{\TP@blockbodyinnersep=#1 \blockbodyinnersep=#1}

\newcommand\useblockstyle[2][]{
    \ifcsname TP@blockstyle#2\endcsname
        \csname TP@blockstyle#2Defaultvalues\endcsname\gpresetkeys{block}{#1}{} % call macro
        \gdef\TP@blockstyle{\csname TP@blockstyle#2\endcsname} % save macro
    \else
        \PackageWarning{tikzposter}{Unknown blockstyle `#2'.}
    \fi
}

 % --------------------------------------- %
 % Block
\newcommand\block[3][]{  % #1 keys   #2 title   #3 text
    % Block position
    \normalsize
    \ifTP@subcolumnEnvironment
        \TP@blocktitlewidth=\subcolwidth
        \TP@blockbodywidth=\subcolwidth
        \TP@blockcenter=\TP@subcolcenter
    \else
        \ifTP@columnEnvironment
            \TP@blocktitlewidth=\colwidth
            \TP@blockbodywidth=\colwidth
            \TP@blockcenter=\TP@colcenter
        \else
            \setlength\TP@blocktitlewidth{\TP@visibletextwidth}
            \setlength\TP@blockbodywidth{\TP@visibletextwidth}
            \TP@blockcenter=0pt
        \fi
    \fi

    % Set parameter
    \setkeys{block}{#1}

    % Settings for Subcolumns
    \ifTP@subcolumnEnvironment \else \ifTP@columnEnvironment \else
        \setlength\TP@blocktitlewidth{\TP@blocktitlewidth-\blocklinewidth}
        \setlength\TP@blockbodywidth{\TP@blockbodywidth-\blocklinewidth}
        \TP@blockcenter=0pt
    \fi \fi

    % Get height of title
    \ifstrempty{#2}{
        \BlockHasTitlefalse
        \setlength{\TP@blocktitleheight}{0pt}
    }{
        \BlockHasTitletrue
        \setbox\TP@blocktitlebox=\hbox{%
            \pgfinterruptpicture%
            \parbox{\TP@blocktitlewidth-2\TP@blocktitleinnersep}{%
                \bf\LARGE\color{blocktitlefgcolor}\TP@blocktitleAlignment#2\par\normalsize}%
            \endpgfinterruptpicture%
        }%
        \setlength{\TP@blocktitleheight}{\ht\TP@blocktitlebox + \dp\TP@blocktitlebox +2\TP@blocktitleinnersep}
    }

    % Get height of body
    \setbox\TP@blockbodybox=\hbox{
        \pgfinterruptpicture%
        \parbox{\TP@blockbodywidth-2\TP@blockbodyinnersep-\TP@blockbodyoffsetx}{%
            \vspace*{\TP@blockbodyverticalshift}\large\color{blockbodyfgcolor}#3\par\normalsize}%
        \endpgfinterruptpicture%
    }%
    \setlength{\TP@blockbodyheight}{\ht\TP@blockbodybox + \dp\TP@blockbodybox +2\TP@blockbodyinnersep}

    % Set coordinates
    \node[minimum width=\TP@blocktitlewidth, minimum height=\TP@blocktitleheight, anchor=center] (blocktitle)%
        at (\TP@blockcenter+\TP@blocktitleoffsetx, {\TP@blocktop-0.5\TP@blocktitleheight+\TP@blocktitleoffsety}){};
    \ifBlockHasTitle
        \node[minimum width=\TP@blockbodywidth, minimum height=\TP@blockbodyheight, anchor=center] (blockbody)%
            at (\TP@blockcenter+\TP@blockbodyoffsetx, {\TP@blocktop-\TP@blocktitleheight-0.5\TP@blockbodyheight+\TP@blockbodyoffsety}){};
    \else
        \node[minimum width=\TP@blockbodywidth, minimum height=\TP@blockbodyheight, anchor=center] (blockbody)%
            at (\TP@blockcenter+\TP@blockbodyoffsetx, {\TP@blocktop-\TP@blocktitleheight-0.5\TP@blockbodyheight}){};
    \fi

    % Draw style
    \TP@blockstyle

    % Draw text
    \ifBlockHasTitle
        \node[text width=\TP@blocktitlewidth-2\TP@blocktitleinnersep, inner sep=\TP@blocktitleinnersep, anchor=center]%
            at (blocktitle){\box\TP@blocktitlebox};
    \fi
    \node[text width=\TP@blockbodywidth-2\TP@blockbodyinnersep, inner sep=\TP@blockbodyinnersep, anchor=center]%
        at (blockbody){\box\TP@blockbodybox};

    % Set position parameter
    \draw let \p1 = (blockbody.south) in node {
        \setlength{\TP@blocktop}{\y1-\TP@blockverticalspace}
        \global\TP@blocktop=\TP@blocktop
    };
    \ifTP@subcolumnEnvironment
        \setlength{\TP@subcolbottom}{\minof{\TP@subcolbottom}{\TP@blocktop}}
        \global\TP@subcolbottom=\TP@subcolbottom
    \else
        \ifTP@columnEnvironment
            \setlength{\TP@colbottom}{\minof{\TP@colbottom}{\TP@blocktop}}
            \global\TP@colbottom=\TP@colbottom
        \fi
    \fi
}

 % --------------------------------------- %
 % Innerblock styles
\gdef\defineinnerblockstyle#1#2#3{
    \expandafter\gdef\csname TP@innerblockstyle#1Defaultvalues\endcsname{\gpresetkeys{innerblock}{#2}{}}
    \expandafter\gdef\csname TP@innerblockstyle#1\endcsname{#3}
}

\define@key{innerblock}{titlewidth}{\setlength{\TP@innerblocktitlewidth}{#1}}
\define@key{innerblock}{bodywidth}{\setlength{\TP@innerblockbodywidth}{#1}}
\define@key{innerblock}{titlewidthscale}{\setlength{\TP@innerblocktitlewidth}{#1\TP@innerblocktitlewidth}}
\define@key{innerblock}{bodywidthscale}{\setlength{\TP@innerblockbodywidth}{#1\TP@innerblockbodywidth}}
\define@boolkey{innerblock}{titleleft}[true]{\ifKV@innerblock@titleleft \gdef\TP@innerblocktitleAlignment{\raggedright} \fi}
\define@boolkey{innerblock}{titlecenter}[true]{\ifKV@innerblock@titlecenter \gdef\TP@innerblocktitleAlignment{\centering} \fi}
\define@boolkey{innerblock}{titleright}[true]{\ifKV@innerblock@titleright \gdef\TP@innerblocktitleAlignment{\raggedleft} \fi}
\define@key{innerblock}{titleoffsetx}{\TP@innerblocktitleoffsetx=#1}
\define@key{innerblock}{titleoffsety}{\TP@innerblocktitleoffsety=#1}
\define@key{innerblock}{bodyoffsetx}{\TP@innerblockbodyoffsetx=#1}
\define@key{innerblock}{bodyoffsety}{\TP@innerblockbodyoffsety=#1}
\define@key{innerblock}{bodyverticalshift}{\TP@innerblockbodyverticalshift=#1}
\define@key{innerblock}{roundedcorners}{\gdef\innerblockroundedcorners{#1}}
\define@key{innerblock}{linewidth}{\innerblocklinewidth=#1}
\define@key{innerblock}{titleinnersep}{\TP@innerblocktitleinnersep=#1 \innerblocktitleinnersep=#1}
\define@key{innerblock}{bodyinnersep}{\TP@innerblockbodyinnersep=#1 \innerblockbodyinnersep=#1}
\define@key{innerblock}{titlebgcolor}{\definecolor{innerblocktitlebgcolor}{named}{#1}}
\define@key{innerblock}{titlefgcolor}{\definecolor{innerblocktitlefgcolor}{named}{#1}}
\define@key{innerblock}{bodybgcolor}{\definecolor{innerblockbodybgcolor}{named}{#1}}
\define@key{innerblock}{bodyfgcolor}{\definecolor{innerblockbodyfgcolor}{named}{#1}}

\newcommand\useinnerblockstyle[2][]{
    \ifcsname TP@innerblockstyle#2\endcsname
        \csname TP@innerblockstyle#2Defaultvalues\endcsname\gpresetkeys{innerblock}{#1}{} % call macro
        \gdef\TP@innerblockstyle{\csname TP@innerblockstyle#2\endcsname} % save macro
    \else
        \PackageWarning{tikzposter}{Unknown innerblockstyle `#2'.}
    \fi
}

 % --------------------------------------- %
 % Innerblock
\newcommand\innerblock[3][]{  % #1 keys   #2 title   #3 text
    % Settings for Innerblock
    \definecolor{innerblocktitlebgcolorTemp}{named}{innerblocktitlebgcolor}
    \definecolor{innerblocktitlefgcolorTemp}{named}{innerblocktitlefgcolor}
    \definecolor{innerblockbodybgcolorTemp}{named}{innerblockbodybgcolor}
    \definecolor{innerblockbodyfgcolorTemp}{named}{innerblockbodyfgcolor}
    \setlength{\TP@innerblocktitlewidth}{\linewidth}
    \setlength{\TP@innerblockbodywidth}{\linewidth}
    \setkeys{innerblock}{#1}

    \begin{tikzpicture}

        % Get center in x-direction
        \pgfmathsetlength{\TP@innerblockcenter}{max(\TP@innerblocktitlewidth,\TP@innerblockbodywidth)/2}

        % Get height of title
        \ifstrempty{#2}{
            \InnerblockHasTitlefalse
            \setlength{\TP@innerblocktitleheight}{0pt}
        }{
            \InnerblockHasTitletrue
            \setbox\TP@innerblocktitlebox=\hbox{%
                \pgfinterruptpicture%
                \parbox{\TP@innerblocktitlewidth-2\TP@innerblocktitleinnersep}{%
                    \bf\color{innerblocktitlefgcolor}\TP@innerblocktitleAlignment#2\par\normalsize}%
                \endpgfinterruptpicture%
                }
            \setlength{\TP@innerblocktitleheight}{\ht\TP@innerblocktitlebox + \dp\TP@innerblocktitlebox +2\TP@innerblocktitleinnersep}
        }

        % Get height of body
        \setbox\TP@innerblockbodybox=\hbox{
            \pgfinterruptpicture%
            \parbox{\TP@innerblockbodywidth-2\TP@innerblockbodyinnersep-\TP@innerblockbodyoffsetx}{%
                \vspace*{\TP@innerblockbodyverticalshift}%
                \color{innerblockbodyfgcolor}#3\par\normalsize}%
            \endpgfinterruptpicture%
        }%
        \setlength{\TP@innerblockbodyheight}{\ht\TP@innerblockbodybox + \dp\TP@innerblockbodybox +2\TP@innerblockbodyinnersep}

        % Set coordinates
        \node[minimum width=\TP@innerblocktitlewidth, minimum height=\TP@innerblocktitleheight, anchor=center] (innerblocktitle)
            at (\TP@innerblockcenter+\TP@innerblocktitleoffsetx, {-0.5\TP@innerblocktitleheight+\TP@innerblocktitleoffsety}) {};%
        \ifInnerblockHasTitle
            \node[minimum width=\TP@innerblockbodywidth, minimum height=\TP@innerblockbodyheight, anchor=center] (innerblockbody)
                at (\TP@innerblockcenter+\TP@innerblockbodyoffsetx, {-\TP@innerblocktitleheight-0.5\TP@innerblockbodyheight+\TP@innerblockbodyoffsety}) {};%
        \else
           \node[minimum width=\TP@innerblockbodywidth, minimum height=\TP@innerblockbodyheight, anchor=center] (innerblockbody)
                at (\TP@innerblockcenter+\TP@innerblockbodyoffsetx, {-\TP@innerblocktitleheight-0.5\TP@innerblockbodyheight}) {};%
        \fi

        % Draw style
        \TP@innerblockstyle

        % Draw text
        \ifInnerblockHasTitle
            \node[text width=\TP@innerblocktitlewidth-2\TP@innerblocktitleinnersep, inner sep=\TP@innerblocktitleinnersep, anchor= center]
                at (innerblocktitle) {\box\TP@innerblocktitlebox};%
        \fi
        \node[text width=\TP@innerblockbodywidth-2\TP@innerblockbodyinnersep, inner sep=\TP@innerblockbodyinnersep, anchor= center]
            at (innerblockbody) {\box\TP@innerblockbodybox};%
    \end{tikzpicture}

    % Reset colors
    \definecolor{innerblocktitlebgcolor}{named}{innerblocktitlebgcolorTemp}
    \definecolor{innerblocktitlefgcolor}{named}{innerblocktitlefgcolorTemp}
    \definecolor{innerblockbodybgcolor}{named}{innerblockbodybgcolorTemp}
    \definecolor{innerblockbodyfgcolor}{named}{innerblockbodyfgcolorTemp}
}

 % --------------------------------------- %
 % Coloredbox
\define@key{coloredbox}{width}{\setlength{\TP@coloredboxwidth}{#1}}
\define@key{coloredbox}{roundedcorners}{\gdef\TP@coloredboxroundedcorners{#1}}
\define@key{coloredbox}{linewidth}{\TP@coloredboxlinewidth=#1}
\define@key{coloredbox}{innersep}{\TP@coloredboxinnersep=#1}
\define@key{coloredbox}{bgcolor}{\definecolor{coloredboxbgcolorTemp}{named}{#1}}
\define@key{coloredbox}{fgcolor}{\definecolor{coloredboxfgcolorTemp}{named}{#1}}
\define@key{coloredbox}{framecolor}{\definecolor{coloredboxframecolorTemp}{named}{#1}}

\newcommand\coloredbox[2][]{  % #1 keys   #2 text
    % Settings for coloredbox
    \setlength{\TP@coloredboxwidth}{\TP@blocktitlewidth-4\TP@blocktitleinnersep}
    \setkeys{coloredbox}{%
     width={\linewidth}, roundedcorners=15, linewidth=3.5pt, innersep=10pt,
bgcolor=notebgcolor, fgcolor=notefgcolor, framecolor=notebgcolor, #1}

    \begin{tikzpicture}
        % Get height of text
        \setbox\TP@coloredbox=\hbox{
            \pgfinterruptpicture%
            \parbox{\TP@coloredboxwidth-2\TP@coloredboxinnersep}{%
                \color{coloredboxfgcolorTemp}#2\par\normalsize}%
            \endpgfinterruptpicture%
        }%
        \setlength{\TP@coloredboxheight}{\ht\TP@coloredbox + \dp\TP@coloredbox +2\TP@coloredboxinnersep}

        % Set coordinates
        \node[minimum width=\TP@coloredboxwidth, minimum height=\TP@coloredboxheight, anchor=center] (coloredbox)
            at (0.5\TP@coloredboxwidth, -0.5\TP@coloredboxheight) {};%

        % Draw style
        \begin{scope}[line width=\TP@coloredboxlinewidth, rounded corners=\TP@coloredboxroundedcorners]
            \draw[color=coloredboxframecolorTemp, fill=coloredboxbgcolorTemp] (coloredbox.south west) rectangle (coloredbox.north east);
        \end{scope}

        % Draw text
        \node[text width=\TP@coloredboxwidth-2\TP@coloredboxinnersep, inner sep=\TP@coloredboxinnersep, anchor= center]
            at (coloredbox) {\box\TP@coloredbox};%
    \end{tikzpicture}

    % Reset colors
    \definecolor{coloredboxbgcolor}{named}{coloredboxbgcolorTemp}
    \definecolor{coloredboxfgcolor}{named}{coloredboxfgcolorTemp}
}

 % --------------------------------------- %
 % Note styles
\gdef\definenotestyle#1#2#3{
    \expandafter\gdef\csname TP@notestyle#1Defaultvalues\endcsname{\gpresetkeys{note}{#2}{}}
    \expandafter\gdef\csname TP@notestyle#1\endcsname{#3}
}

\define@key{note}{targetoffsetx}{\TP@notetargetoffsetx=#1}
\define@key{note}{targetoffsety}{\TP@notetargetoffsety=#1}
\define@key{note}{angle}{\def\TP@noteangle{#1}}
\define@key{note}{radius}{\TP@noteradius=#1}
\define@key{note}{width}{\TP@notewidth=#1}
\define@boolkey{note}{connection}[true]{\NoteHasConnectiontrue \ifKV@note@connection \NoteHasConnectiontrue \else \NoteHasConnectionfalse \fi}
\define@key{note}{rotate}{\gdef\noterotate{#1}}
\define@key{note}{roundedcorners}{\gdef\noteroundedcorners{#1}}
\define@key{note}{linewidth}{\notelinewidth=#1}
\define@key{note}{innersep}{\TP@noteinnersep=#1 \noteinnersep=#1}
\newcommand\usenotestyle[2][]{
    \ifcsname TP@notestyle#2\endcsname
        \csname TP@notestyle#2Defaultvalues\endcsname\gpresetkeys{note}{#1}{} % call macro
        \gdef\TP@notestyle{\csname TP@notestyle#2\endcsname} % save macro
    \else
        \PackageWarning{tikzposter}{Unknown notestyle `#2'.}
    \fi
}

 % --------------------------------------- %
 % Note
\newcommand\note[2][]{  % #1 keys   #2 text
    \normalsize
    \setkeys{note}{#1}

    % Note box
    \setbox\TP@notebox=\hbox{%
        \pgfinterruptpicture%
        \parbox{\TP@notewidth-2\TP@noteinnersep}{\color{notefgcolor}#2\par\normalsize}%
        \endpgfinterruptpicture%
    }%
    \setlength{\TP@noteheight}{\ht\TP@notebox + \dp\TP@notebox +2\TP@noteinnersep}
    \pgfmathsetlength{\TP@noteheight}{max(\TP@noteheight,80pt)}

    % Set note coordinates
    \node (notetarget) at ($(blockbody)+(\TP@notetargetoffsetx,\TP@notetargetoffsety)$){};
    \node[minimum width=\TP@notewidth, minimum height=\TP@noteheight, anchor=center,rotate=\noterotate] (notecenter)
        at ($(notetarget)+({\TP@noteradius*cos(\TP@noteangle)},{\TP@noteradius*sin(\TP@noteangle)})$){};%

    % Draw note
    \begin{pgfonlayer}{notelayer}
        \TP@notestyle
        % Draw note text
        \node[text width=\TP@notewidth-2\TP@noteinnersep, inner sep=\TP@noteinnersep, anchor=center,rotate=\noterotate]%
            at (notecenter){\box\TP@notebox};
    \end{pgfonlayer}
}

 % --------------------------------------- %
 % Color palette
\gdef\definecolorpalette#1#2{\expandafter\gdef\csname TP@colorpalette#1\endcsname{#2}}

\gdef\usecolorpalette#1{
    \ifcsname TP@colorpalette#1\endcsname
         \csname TP@colorpalette#1\endcsname % call macro
    \else
        \PackageWarning{tikzposter}{Unknown color palette `#1'.}
    \fi
}

 % --------------------------------------- %
 % Color style
\gdef\definecolorstyle#1#2#3{
    \expandafter\gdef\csname TP@colorstyle#1Defaultvalues\endcsname{#2}
    \expandafter\gdef\csname TP@colorstyle#1\endcsname{#3}
}

\define@key{colors}{colorOne}{\colorlet{colorOne}{#1}}
\define@key{colors}{colorTwo}{\colorlet{colorTwo}{#1}}
\define@key{colors}{colorThree}{\colorlet{colorThree}{#1}}
\define@key{colors}{colorPalette}{\usecolorpalette{#1}}

\newcommand\usecolorstyle[2][]{  % #1 keys   #2 macroname
    \ifcsname TP@colorstyle#2\endcsname
         \csname TP@colorstyle#2Defaultvalues\endcsname % call macro
         \setkeys{colors}{#1} % set keys
         \csname TP@colorstyle#2\endcsname % call macro
    \else
        \PackageWarning{tikzposter}{Unknown color style `#2'.}
    \fi
}

 % --------------------------------------- %
 % Layouttheme
\gdef\definelayouttheme#1#2{\expandafter\gdef\csname TP@layouttheme#1\endcsname{#2}}

\gdef\usetheme#1{
    % Layout theme
    \ifcsname TP@layouttheme#1\endcsname
         \csname TP@layouttheme#1\endcsname % call macro
    \else
        \PackageWarning{tikzposter}{Unknown layouttheme `#1'.}
    \fi
}

 % Input Style and Theme Files
\input{tikzposterColorpalettes}
\input{tikzposterColorstyles}
\input{tikzposterBackgroundstyles}
\input{tikzposterTitlestyles}
\input{tikzposterBlockstyles}
\input{tikzposterInnerblockstyles}
\input{tikzposterNotestyles}
\input{tikzposterLayoutthemes}

 % Set Default theme
\usetheme{Default}

 % --------------------------------------- %
 % Columns environment
\newenvironment{columns}{
    \TP@columnEnvironmenttrue
    \setlength{\TP@colcenter}{-0.5\TP@visibletextwidth-\TP@colspace-0.5\blocklinewidth}
    \global\TP@colcenter=\TP@colcenter
    \global\TP@coltop=\TP@blocktop
    \global\TP@colbottom=\TP@blocktop
    \colwidth=0pt
}{
    \TP@columnEnvironmentfalse
    \global\TP@blocktop=\TP@colbottom
}

 % Column
\gdef\column#1{  % #1: relative width
    \ifTP@columnEnvironment
        \normalsize
        \setlength{\TP@blocktop}{\TP@coltop}
        \setlength{\TP@colcenter}{\TP@colcenter+0.5\colwidth+\TP@colspace}
        \setlength{\colwidth}{#1\TP@visibletextwidth+#1\TP@colspace-\TP@colspace-\blocklinewidth}
        \setlength{\TP@colcenter}{\TP@colcenter+0.5\colwidth+\blocklinewidth}
    \fi
}

 % --------------------------------------- %
 % Subcolumns environment
\newenvironment{subcolumns}{
    \ifTP@columnEnvironment
        \TP@subcolumnEnvironmenttrue
        \setlength{\TP@subcolcenter}{\TP@colcenter-0.5\colwidth-\TP@blockbodyinnersep-\TP@subcolspace-\TP@blockbodyinnersep}
        \global\TP@subcolcenter=\TP@subcolcenter
        \global\TP@subcoltop=\TP@blocktop
        \global\TP@subcolbottom=\TP@blocktop
        \subcolwidth=0pt
    \fi
}{
    \TP@subcolumnEnvironmentfalse
    \global\TP@blocktop=\TP@subcolbottom
}

 % Subcolumn
\gdef\subcolumn#1{  % #1: relative width
    \ifTP@subcolumnEnvironment
        \normalsize
        \setlength{\TP@blocktop}{\TP@subcoltop}
        \setlength{\TP@subcolcenter}{\TP@subcolcenter+0.5\subcolwidth+\TP@blockbodyinnersep+\TP@subcolspace+\TP@blockbodyinnersep}
        \setlength{\subcolwidth}{#1\colwidth+#1\TP@blockbodyinnersep+#1\TP@blockbodyinnersep
            +#1\TP@subcolspace-\TP@subcolspace-2\TP@blockbodyinnersep}
        \setlength{\TP@subcolcenter}{\TP@subcolcenter+0.5\subcolwidth}
    \fi
}

 % --------------------------------------- %
 % Affection
\gdef\tikzposterlatexaffectionproofon{\TP@showlatexaffectiontrue}
\gdef\tikzposterlatexaffectionproofoff{\TP@showlatexaffectionfalse}
\TP@showlatexaffectiontrue

 % --------------------------------------- %
 % Document environment
\AtBeginDocument{%
    % Get titlegraphic height
    \settototalheight{\titlegraphicheight}{\hbox{\@titlegraphic}}

    \begin{center}
    \begin{tikzpicture}
        % Background
        \coordinate (topright) at (0.5\textwidth, 0.5\textheight);
    \coordinate (bottomleft) at (-0.5\textwidth, -0.5\textheight);
    \clip (bottomleft) rectangle (topright);

    % Draw background
    \begin{pgfonlayer}{backgroundlayer}
        \clip (bottomleft) rectangle (topright);
        \TP@backgroundstyle
        \ifTP@showlatexaffection
                \node[inner sep=4pt, anchor=south east, fill=white, draw=none, rounded corners=5, fill opacity=0.3, text opacity=1]
                    at (0.5\textwidth-7pt, -0.5\textheight+7pt){\footnotesize {\bf\textrm\LaTeX}~\textrm{Ti\emph{k}Z}\bf\textrm{poster}};
            \fi
    \end{pgfonlayer}

}

\AtEndDocument{%
        \end{tikzpicture}
    \end{center}
}

%%

\newcounter{figurecounter}

%% #1 Caption
\newenvironment{tikzfigure}[1][]{
  \def \rememberparameter{#1}
  \vspace{10pt}
  \refstepcounter{figurecounter}
  \begin{center}
  }{
    \ifx\rememberparameter\@empty
    \else %nothing
    \\[10pt]
    {\small Fig.~\thefigurecounter: \rememberparameter}
    \fi
  \end{center}
}

%%



\endinput
%%
%% End of file `tikzposter.cls'.
