\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iiuwb}[2011/11/26 Institute of Informatics of UwB LaTeX class]
\DeclareOption*{% wszystkie opcje
\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
\LoadClass[12pt,a4paper,oneside]{report}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{array,graphicx,ifthen,float}

%Komendy, które muszą być nadpisane przez autora
\DeclareRobustCommand{\imiona}{Imię}
\DeclareRobustCommand{\nazwiska}{Nazwisko}

\DeclareRobustCommand{\stopienpromotora}{dr}
\DeclareRobustCommand{\imionapromotora}{Imię promotora}
\DeclareRobustCommand{\nazwiskapromotora}{Nazwiska promotora}
\DeclareRobustCommand{\profuwb}{TAK}


\DeclareRobustCommand{\stopienasystenta}{dr}
\DeclareRobustCommand{\imionaasystenta}{Asystent}
\DeclareRobustCommand{\nazwiskaasystenta}{Asystent}

\DeclareRobustCommand{\rodzajpracy}{PRACA LICENCJACKA}
\DeclareRobustCommand{\rok}{2022}
\DeclareRobustCommand{\tytul}{Tytuł}
\DeclareRobustCommand{\zaklad}{Zakład}

\DeclareRobustCommand{\stronatytulowa}{
\begin{titlepage}\Large
\begin{center}
UNIWERSYTET W BIAŁYMSTOKU\\
WYDZIAŁ INFORMATYKI
\end{center}
\vfill
\begin{figure}[!h]
\centering
\includegraphics[width=6cm]{uwb.jpg}
\end{figure}
\vfill
\begin{center}
{\large \imiona{} \MakeUppercase{\nazwiska}}\\
\end{center}
\begin{center}
{\LARGE \textbf{\tytul}}
\end{center}

\begin{center}
{\large \textbf{\rodzajpracy}}
\end{center}
\vfill
\begin{flushright}
\begin{tabular}{r}

Promotor:\\
\stopienpromotora\ \imionapromotora{} \MakeUppercase{\nazwiskapromotora}\ifthenelse{\equal{\profuwb}{TAK}}{, prof. UwB}{}
\\

\ifthenelse{\NOT{\equal{\nazwiskaasystenta}{}}}{
Asystent:\\
\stopienasystenta\ \imionaasystenta{} \MakeUppercase{\nazwiskaasystenta}\\
}{}
\end{tabular}
\end{flushright}
\begin{center}
BIAŁYSTOK \rok
\end{center}
\end{titlepage}
}

\AtBeginDocument{%
\renewcommand{\baselinestretch}{1.3}
\newcommand{\source}[1]{\caption*{Źródło {#1}} }
\stronatytulowa\newpage
\setcounter{page}{2}
}

%POZIOM
\setlength{\hoffset}{-17mm}
\setlength{\oddsidemargin}{25mm}
\setlength{\evensidemargin}{25mm}
\setlength{\textwidth}{150mm}
\setlength{\marginparsep}{0mm}
\setlength{\marginparwidth}{0mm}

%PION
\setlength{\voffset}{-1in}
\setlength{\topmargin}{25mm}
\setlength{\headheight}{0mm}
\setlength{\headsep}{0mm}
\setlength{\textheight}{247mm}
\setlength{\footskip}{15mm}

\endinput
