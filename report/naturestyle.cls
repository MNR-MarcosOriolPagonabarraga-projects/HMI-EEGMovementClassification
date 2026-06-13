\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{naturestyle}[2026/06/09 Custom Nature-style Journal Class]

% Pass standard options to the article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt,twocolumn,a4paper]{article}

% --- Geometry & Layout ---
\RequirePackage[
  margin=0.72in,
  top=1in,         % Pushes the main text body slightly higher up
  headheight=1.2cm,  % Gives the logo a dedicated bounding box height
  headsep=0.4in,
  columnsep=0.26in
]{geometry}
\RequirePackage{microtype} 

% --- Typography ---
\RequirePackage{newtxtext,newtxmath} 
\RequirePackage[scaled]{helvet}

% --- Required Packages ---
\RequirePackage{graphicx} % Explicitly loaded here
\RequirePackage{fancyhdr}  % Explicitly loaded here
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{xcolor}
\RequirePackage{hyperref}

% --- Link Styling ---
\definecolor{natureblue}{RGB}{0, 50, 100}
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=natureblue,
    urlcolor=natureblue
}

% --- Header/Footer Setup (fancyhdr) ---
\pagestyle{fancy}
\fancyhf{} % Clear default settings

% Styling for pages 2 and onwards
\renewcommand{\headrulewidth}{0.4pt} 
\fancyhead[L]{\sffamily\small\color{gray} HMI Final Project Report}
\fancyhead[R]{\includegraphics[height=1.2cm,keepaspectratio]{logo-ETSEIB-upc.png}}

% Define a special style JUST for the title page
\fancypagestyle{titlepage}{
  \fancyhf{} % Clear headers and footers for the first page
  \renewcommand{\headrulewidth}{0pt} % REMOVE the horizontal line
  % Keep ONLY the logo in the top right, slightly lower if needed
  \fancyhead[R]{\includegraphics[height=1.4cm,keepaspectratio]{logo-ETSEIB-upc.png}}
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

% --- Custom Maketitle ---
\renewcommand{\maketitle}{
  \twocolumn[
    \begin{center}
      \vspace*{1.5em}
      % Use \sffamily (Helvetica), \huge, and \bfseries for the Nature style title
      {\fontfamily{phv}\fontseries{b}\fontsize{22pt}{26pt}\selectfont \@papertitle \par}
      \vspace{1.8em}
      {\sffamily\large \@paperauthor \par}
      \vspace{0.4em}
      {\sffamily\small\color{gray} \@paperaffil \par}
      \vspace{2.5em}
    \end{center}
    
    % Nature style abstract: bold, sans-serif, acting as the first paragraph
    \bgroup
    \sffamily\small\bfseries
    \@paperabstract
    \egroup
    
    \vspace{2.5em}
  ]
  \thispagestyle{titlepage} % Keep the line-free logo layout on page 1
}