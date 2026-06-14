\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{naturestyle}[2026/06/13 Premium Journal Class]

% Pass standard options to the article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt,twocolumn,a4paper]{article}

% --- Geometry & Layout ---
\RequirePackage[
  margin=0.72in,
  top=1.2in,         
  headheight=1.5cm,  % Expanded to fix the fancyhdr UPC logo warning
  headsep=0.4in,
  columnsep=0.26in
]{geometry}
\RequirePackage{microtype} 

% --- Typography ---
% Upgraded from Times to Palatino for a premium, book-like mathematical aesthetic
\RequirePackage{newpxtext,newpxmath} 
\RequirePackage[scaled=0.92]{helvet}

% --- Required Packages ---
\RequirePackage{graphicx} 
\RequirePackage{fancyhdr}  
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{xcolor}
\RequirePackage{hyperref}

% --- Link Styling ---
\definecolor{journalblue}{RGB}{0, 52, 110} % Slightly deeper, richer blue
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=journalblue,
    urlcolor=journalblue
}

% --- Header/Footer Setup (fancyhdr) ---
\pagestyle{fancy}
\fancyhf{} 

% Styling for pages 2 and onwards
\renewcommand{\headrulewidth}{0.4pt} 
\fancyhead[L]{\sffamily\small\color{gray} HMI Final Project Report}
\fancyhead[R]{\includegraphics[height=1.2cm,keepaspectratio]{logo-ETSEIB-upc.png}}

% Define a special style JUST for the title page
\fancypagestyle{titlepage}{
  \fancyhf{} 
  \renewcommand{\headrulewidth}{0pt} 
  \fancyhead[L]{\raisebox{-15mm}{\includesvg[height=1.6cm,keepaspectratio]{etseib_logo}}}
}

% --- Section Headings ---
\titleformat{\section}
  {\sffamily\large\bfseries\color{black}}
  {\thesection.}
  {0.5em}
  {}
\titleformat{\subsection}
  {\sffamily\normalsize\bfseries\color{black}}
  {\thesubsection.}
  {0.5em}
  {}
  
% --- Caption Styling ---
\captionsetup{
  labelfont={sf,bf},
  textfont=sf,
  font=small,
  labelsep=space
}

% --- Title Block Variables ---
\newcommand{\@papertitle}{}
\newcommand{\papertitle}[1]{\renewcommand{\@papertitle}{#1}}

\newcommand{\@paperauthor}{}
\newcommand{\paperauthor}[1]{\renewcommand{\@paperauthor}{#1}}

\newcommand{\@paperaffil}{}
\newcommand{\paperaffil}[1]{\renewcommand{\@paperaffil}{#1}}

\newcommand{\@paperabstract}{}
\newcommand{\paperabstract}[1]{\renewcommand{\@paperabstract}{#1}}

% --- Custom Maketitle (Top-Tier Modern Layout) ---
\renewcommand{\maketitle}{
  \twocolumn[
    \begin{flushleft} 
      {\fontfamily{phv}\fontseries{b}\fontsize{24pt}{28pt}\selectfont \@papertitle \par}
      \vspace{0.8em}
      {\sffamily\large \@paperauthor \par}
      \vspace{0.2em}
      {\sffamily\small\color{darkgray} \@paperaffil \par}
      \vspace{0.8em} % Tighter space before abstract rule
      % Top Abstract Rule
      \noindent\rule{\textwidth}{0.8pt} 
    \end{flushleft}
    
    % Abstract Block
    \bgroup
    \sffamily\small\bfseries
    \noindent \@paperabstract
    \egroup
    
    \begin{flushleft}
      \vspace{0.3em}
      % Bottom Abstract Rule
      \noindent\rule{\textwidth}{0.8pt} 
      \vspace{1em} % Tighter space before the main text starts
    \end{flushleft}
  ]
  \thispagestyle{titlepage} 
}