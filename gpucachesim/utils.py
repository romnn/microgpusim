import typing
import click
import platform
import pyperclip
import numpy as np
from pathlib import Path
from os import PathLike


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


def flatten(l):
    return [item for ll in l for item in ll]


def dedup(l):
    return list(dict.fromkeys(l))


def two_closest_divisors(n):
    assert isinstance(n, int)
    a = int(np.round(np.sqrt(n)))
    while n % a > 0:
        a -= 1
    return a, n // a


def round_up_to_next_power_of_two(x):
    exp = np.ceil(np.log2(x)) if x > 0 else 1
    return np.power(2, exp)


def round_down_to_next_power_of_two(x):
    exp = np.floor(np.log2(x)) if x > 0 else 1
    return np.power(2, exp)


def round_to_multiple_of(x, multiple_of):
    return multiple_of * np.round(x / multiple_of)


def round_up_to_multiple_of(x, multiple_of):
    return multiple_of * np.ceil(x / multiple_of)


def round_down_to_multiple_of(x, multiple_of):
    return multiple_of * np.floor(x / multiple_of)


def copy_to_clipboard(value):
    try:
        pyperclip.copy(value)
    except pyperclip.PyperclipException as e:
        print("copy to clipboard failed: {}".format(e))


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
"""


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


MINUTES = 60


def convert_to_png(
    input_path: PathLike,
    output_path: PathLike,
    max_size: typing.Optional[int] = 4096 * 2,
    quality=100,
    density=300,
    timeout_sec=8 * MINUTES,
):
    import gpucachesim.cmd as cmd_utils

    input_path = Path(input_path)
    assert input_path.is_file()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "convert",
    ]
    if isinstance(density, int):
        cmd += [
            "-density",
            str(density),
        ]
    cmd += [str(input_path)]

    if isinstance(quality, int):
        cmd += [
            "-quality",
            str(quality),
        ]
    if isinstance(max_size, int):
        cmd += [
            "-resize",
            "{}x{}".format(max_size, max_size),
        ]
    cmd += [str(output_path)]
    cmd = " ".join(cmd)
    _, stdout, stderr, _ = cmd_utils.run_cmd(
        cmd,
        timeout_sec=timeout_sec,
    )


@main.command()
@click.option("-i", "--input", "input_path", help="path to input file")
@click.option("-o", "--output", "output", help="path to output file")
@click.option("-s", "--size", "max_size", help="max size in any dimension")
@click.option("-q", "--quality", "quality", help="quality")
@click.option("-d", "--density", "density", help="density")
def to_png(input_path, output_path, max_size, quality, density):
    convert_to_png(
        input_path=input_path,
        output_path=output_path,
        max_size=max_size,
        quality=quality,
        density=density,
    )
    # import gpucachesim.cmd as cmd_utils

    # cmd = [
    #     "convert",
    #     "-density",
    #     str(density),
    #     str(input_path),
    #     "-quality",
    #     str(quality),
    #     "-resize",
    #     "{}x{}".format(max_size, max_size),
    #     str(output_path),
    # ]
    # cmd = " ".join(cmd)
    # _, stdout, stderr, _ = cmd_utils.run_cmd(
    #     cmd,
    #     timeout_sec=1 * 60,
    # )
    # print(stdout)
    # print(stderr)


if __name__ == "__main__":
    main()
    # main(ctx={})
