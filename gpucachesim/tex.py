from pathlib import Path
from os import PathLike

TEX_PACKAGES = r"""
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[table]{xcolor}

\usepackage[utf8]{inputenc} % Required for inputting international characters
\usepackage[scaled]{helvet}
\usepackage[T1]{fontenc} % Output font encoding for international characters
\renewcommand\familydefault{\sfdefault}

\usepackage{graphicx}
\usepackage{tabulary}
\usepackage{tabularx}
\usepackage{listings}

% \usepackage{lstlinebgrd}
% listings-rust}
\usepackage[norndcorners,customcolors]{hf-tikz}
\hfsetbordercolor{yellow}
\hfsetfillcolor{yellow}

\usepackage{subcaption}
\usepackage{multirow}
\usepackage{multicol}
\usepackage{makecell}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{layouts}

% font for checkmark ding symbols
\usepackage{pifont}
\usepackage{makecell}

% for command line bash code
\usepackage{minted}

% options in square brackets for enumerate and itemize
\usepackage{enumitem}

% fix table row colors
\newcounter{tblerows}
\expandafter\let\csname c@tblerows\endcsname\rownum

% make big, small and tiny column width for tabularx
\newcolumntype{b}{>{\raggedright\arraybackslash}X}
% \newcolumntype{s}{>{\raggedright\arraybackslash\hsize=.6\hsize}X}
% \newcolumntype{d}{>{\raggedright\arraybackslash\hsize=.35\hsize}X}
% \newcolumntype{t}{>{\raggedright\arraybackslash\hsize=.22\hsize}X}
% centered version of t
% \newcolumntype{u}{>{\centering\arraybackslash\hsize=.22\hsize}X}
% right aligned version of s (small)
\newcolumntype{z}{>{\raggedleft\arraybackslash\hsize=.6\hsize}X}
% right aligned version of s (super small)
\newcolumntype{s}{>{\raggedleft\arraybackslash\hsize=.3\hsize}X}
% right aligned version of s (large)
\newcolumntype{Z}{>{\raggedleft\arraybackslash}X}
% centered aligned version of s
\newcolumntype{k}{>{\centering\arraybackslash\hsize=.6\hsize}X}

\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
\newcommand*\xor{\oplus}

\newcommand\todo[1]{\textcolor{red}{TODO: #1}}

% \newcommand\descitem[1]{\item{\bfseries #1}\\}
\newcommand\descitem[1]{\item{\bfseries #1}}

% name of the simulator
% \newcommand{\simName}{\textsc{microgpusim}~}
\newcommand{\simName}{\textsc{microgpusim}}
"""


def col(align="c", hsize=None, width=None, base=None):
    col = ""
    valid_hsize = 1.0
    if isinstance(hsize, float):
        valid_hsize = hsize
    match align.lower().strip():
        case "c" | "center":
            col += r">{\hsize=" + str(valid_hsize) + r"\hsize}"
        case "r" | "right":
            col += r">{\raggedleft\arraybackslash"
            col += r"\hsize=" + str(valid_hsize) + r"\hsize}"
        case "l" | "left":
            col += r">{\raggedright\arraybackslash"
            col += r"\hsize=" + str(valid_hsize) + r"\hsize}"
    if base is not None:
        col += str(base)
    elif width is not None:
        col += "p{" + str(width) + "}"
    else:
        col += "X"
    return col


def r(**kwargs):
    return col(align="right", **kwargs)


def sanitize_caption(caption) -> str:
    valid_lines = [line for line in caption.splitlines() if line.strip() != ""]
    return "\n".join(valid_lines)


def render_latex(tex_code: str, output_path: PathLike, crop=True):
    import gpucachesim.cmd as cmd_utils
    import tempfile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # write code to temp dir
        tex_input_file = temp_dir / "code.tex"
        with open(tex_input_file, "w") as f:
            f.write(tex_code)

        # run pdflatex
        cmd = [
            "pdflatex",
            "-output-directory",
            str(temp_dir),
            "-interaction",
            "nonstopmode",
            "-shell-escape",
            str(tex_input_file),
        ]
        try:
            retcode, stdout, stderr, _ = cmd_utils.run_cmd(
                cmd,
                timeout_sec=60,
                cwd=temp_dir,
            )
        except cmd_utils.ExecStatusError as e:
            print("stderr:")
            print(e.stderr)
            print("stdout:")
            print(e.stdout)
            tex_log_file = tex_input_file.with_suffix(".log")
            with open(tex_log_file, "r") as f:
                tex_log = f.read()
                # print(tex_log)
            # raise e

        # if ret_code != 0:
        #     print("stdout:")
        #     print(stdout)
        #     print("stderr:")
        #     print(stderr)
        #     raise ValueError("cmd {} failed with code {}", cmd, ret_code)

        # todo: read the log file of pdflatex...
        # todo: copy the resulting pdf file to the output

        tex_output_pdf = tex_input_file.with_suffix(".pdf")
        assert tex_output_pdf.is_file()

        if crop:
            cmd = ["pdfcrop", str(tex_output_pdf), str(tex_output_pdf)]
            cmd_utils.run_cmd(
                " ".join(cmd),
                timeout_sec=60,
                cwd=temp_dir,
            )

        assert tex_output_pdf.is_file()
        tex_output_pdf.rename(output_path)
