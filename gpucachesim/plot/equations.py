import click
import gpucachesim.plot as plot
import gpucachesim.utils as utils
from wasabi import color


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


@main.command()
@click.option("--png", "png", type=bool, default=True, help="convert to png")
def equations(png):
    equations = [
        #         (
        #             "amdahl_n",
        #             r"""
        # \begin{equation}
        # n=8
        # \end{equation}
        #         """,
        #         ),
        (
            "amdahl",
            # 1 / ((1 - p) + p / n)
            r"""
\begin{align*}
S_{\text{Amdahl}}(p, n) = \frac{1}{1 - (1-p) + \frac{p}{n}}
\end{align*}
        """,
        ),
        (
            "gustafson",
            # 1 + (n - 1) * p
            r"""
\begin{align*}
S_{\text{Gustafson}}(p, n) = 1 + (n-1)p
\end{align*}
        """,
        ),
        (
            "amdahl_p_90",
            r"""
\begin{align*}
S_{\text{Amdahl}}(0.90, 8) =  4.8
\end{align*}
""",
        ),
        (
            "amdahl_p_83",
            r"""
\begin{align*}
S_{\text{Amdahl}}(0.83, 8) = 3.6
\end{align*}
        """,
        ),
        (
            "gustafson_p_90",
            r"""
\begin{align*}
S_{\text{Gustafson}}(0.90, 8) = 7.3
\end{align*}
        """,
        ),
        (
            "gustafson_p_83",
            r"""
\begin{align*}
S_{\text{Gustafson}}(0.83, 8) = 6.8
\end{align*}
        """,
        ),
        (
            "cache_capacity",
            r"""
\begin{align*}
C = S \cdot a \cdot b
\end{align*}
        """,
        ),
        (
            "cache_parameters",
            r"""
\begin{align*}
C &= \text{Cache capacity} \\
S &= \text{Cache sets} \\
a &= \text{Cache associativity (way size)} \\
b &= \text{Cache line (block) size}
\end{align*}
        """,
        ),
        (
            "offset_bit_0",
            r"\begin{align*}"
            + "\n"
            + r"\Delta_0(a)=& (a_{10} \lor a_{12} \lor a_{14}) "
            + r"      \land (a_{10} \lor \neg a_{12} \lor \neg a_{14}) "
            + r"      \land (a_{12} \lor \neg a_{10} \lor \neg a_{14}) "
            + r"      \land (a_{14} \lor \neg a_{10} \lor \neg a_{12}) \\"
            + r"    & \land (\neg a_{11} \lor \neg a_{13} \lor \neg a_{14}) "
            + r"      \land (\neg a_{12} \lor \neg a_{13} \lor \neg a_{14}) "
            + r"      \land (\neg a_{13} \lor \neg a_{14} \lor \neg a_{9}) "
            + "\n"
            + r"\end{align*}",
        ),
        # (b10 | b12 | b14) & (b10 | ~b12 | ~b14) & (b12 | ~b10 | ~b14) & (b14 | ~b10 | ~b12) & (~b11 | ~b13 | ~b14) & (~b12 | ~b13 | ~b14) & (~b13 | ~b14 | ~b9)
        (
            "offset_bit_1",
            r"\begin{align*}"
            + "\n"
            + r"\Delta_1(a)=(a_{11} \land a_{13} \land \neg a_{9}) "
            + r"\lor (a_{11} \land a_{9} \land \neg a_{13}) "
            + r"\lor (a_{13} \land a_{9} \land \neg a_{11}) "
            + r"\lor (\neg a_{11} \land \neg a_{13} \land \neg a_{9}) "
            + "\n"
            + r"\end{align*}",
        ),
        # (b11 & b13 & ~b9) | (b11 & b9 & ~b13) | (b13 & b9 & ~b11) | (~b11 & ~b13 & ~b9
        (
            "offset_bit_xor",
            r"\begin{align*}"
            + "\n"
            + r"\Delta_0(a)&=a_{10} \xor a_{12} \xor a_{14} \\"
            + r"\Delta_1(a)&=\neg \Delta_0(a) \xor a_{9} \xor a_{10} \xor a_{11} \xor a_{12} \xor a_{13} \xor a_{14}"
            + "\n"
            + r"\end{align*}",
        ),
        (
            "trace_formats",
            r"""
\begin{table}[htbp]
\fontsize{8}{10}\selectfont
\footnotesize
\caption{\small Comparison of the JSON, MessagePack, and Accelsim trace instruction serialization formats w.r.t. size and deserialization speed. The unit KI corresponds to kilo (1000) trace instructions.}\label{table:trace-format-comparison}
\centering
\setlength\extrarowheight{2pt}
% \rowcolors{3}{white}{gray!20}
{\renewcommand{\arraystretch}{1.5}%
\begin{tabularx}{0.6\textwidth}{Z|s|z|z}
% ---- paste here
& Size & \multicolumn{2}{c}{Deserialization speed} \\
 & MiB/KI & MiB/sec & KI/sec \\ \hline
\rowcolor{gray!10}Accelsim & 0.24 & 64.35 & 320.24 \\
Json & 1.52 & 502.79 & 331.29 \\
\rowcolor{gray!10}MessagePack & 0.14 & 180.42 & 1257.99 \\
% ---- paste end
\end{tabularx}}
\end{table}
""",
        ),
        (
            "latency_distribution_configurations",
            r"""
\begin{table}[htbp]
\fontsize{8}{10}\selectfont
\footnotesize
\caption{\small Configurations of the fine-grained p-chase microbenchmark used to collect the full latency distribution for the NVIDIA TitanX (Pascal).}\label{table:latency-distribution-parameters}
\centering
\setlength\extrarowheight{2pt}
% \rowcolors{3}{white}{gray!20}
{\renewcommand{\arraystretch}{1.5}%
\begin{tabularx}{0.9\textwidth}{Z|z|Z|z|z|z}
& \multicolumn{1}{c}{L1}
& \multicolumn{1}{c}{$N$}
& \multicolumn{1}{c}{$s$}
& \multicolumn{1}{c}{$w$}
& \multicolumn{1}{c}{$r$} \\ \hline
\rowcolor{gray!10}
Configuration 1 & \cmark & $256\text{B}$ ($\ll C_{\text{L1}}$) & $4\text{B}$ & $1$ & $1$ \\
Configuration 2 & \xmark & $256\text{B}$ ($\ll C_{\text{L1}}$) & $4\text{B}$ & $1$ & $1$ \\
\rowcolor{gray!10}
Configuration 3 & \xmark & $6\text{MiB}$ ($\ll C_{\text{L2}}$) & $128\text{B}$ & $0$ & $1$ \\
\end{tabularx}}
\end{table}
""",
        ),
        (
            "matrixmul_generic",
            r"""
\begin{align*}
A^{m \times n} \times B^{n \times p} =
\begin{bmatrix}
a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4} & \cdots & a_{1,n} \\
a_{2,1} & a_{2,2} & a_{2,3} & a_{2,4} & \cdots & a_{2,n} \\
\vdots & \vdots &  \vdots &  \vdots & \ddots & \vdots \\
a_{m,1} & a_{m,2} & a_{m,3} & a_{m,4} & \cdots & a_{m,n}
\end{bmatrix}
\times
\begin{bmatrix}
b_{1,1} & b_{1,2} & \cdots & b_{1,p} \\
b_{2,1} & b_{2,2} & \cdots & b_{2,p} \\
b_{3,1} & b_{3,2} & \cdots & b_{3,p} \\
b_{4,1} & b_{4,2} & \cdots & b_{4,p} \\
\vdots & \vdots & \ddots & \vdots \\
b_{n,1} & b_{n,2} & \cdots & b_{n,p}
\end{bmatrix}
\end{align*}
""",
        ),
        (
            "gto_scheduler",
            r"""
\begin{algorithm}
\caption{Greedy-then-oldest warp scheduler}
\begin{algorithmic}
\Require Warps $W \gets \{w_{1},..,w_{N}\}$ assigned to warp scheduler $s$.
\Require Last issued warp $w'\in\emptyset \cup W$ assigned to warp scheduler $s$.
\State $W' \gets$ sorted($w$) (oldest warps first)
\State $W' \gets w'\cup W'$ 
% \State last\_issued $\gets \emptyset$
\State issued $\gets 0$
\For{$w \in W$}
\While{$w$.has\_instruction() and not $w$.waiting() and not $w$.at\_barrier()}
\If{issued $\geq 2$}
\State \textbf{break} \Comment{can issue up to 2 instructions per cycle}
\EndIf
\State instruction $\gets$ get\_warp\_instruction(warp)
\If{scoreboard.has\_collision(instruction)}
\State \textbf{continue}
\EndIf
\State exec\_unit $\gets$ get\_exec\_unit(instruction)
\If{not already\_issued\_to(exec\_unit) and can\_issue(instruction, exec\_unit)}
\State issue(instruction, exec\_unit)
\State issued++
\State $w' \gets \{w\}$
\EndIf
\EndWhile
\If{issued $> 0$}
\State \textbf{return} \Comment{instructions must come from same warp}
\EndIf
\EndFor
\end{algorithmic}
\end{algorithm}
""",
        ),
        (
            "simulator_config_parameters",
            r"""
\begin{table}[htbp]
\fontsize{8}{10}\selectfont
\footnotesize
\caption{\small Simulator configuration parameters.}
\label{table:validation-simulator-config}
\centering
\setlength\extrarowheight{2pt}
% \rowcolors{3}{white}{gray!20}
{\renewcommand{\arraystretch}{1.5}%
\begin{tabularx}{\textwidth}{ZX}
\textbf{mode} & serial, deterministic, nondeterministic \\
\textbf{memory only} & \lstinline{true}, \lstinline{false} \\
\textbf{clusters} & 28, 112 \\
\textbf{cores per cluster} & 1, 4 \\
\textbf{threads} (mode $\neq$ serial) & 4, 8 \\
\textbf{run ahead} (mode $=$ nondeterministic) & 5, 10 \\
\end{tabularx}}
\end{table}
""",
        ),
        (
            "benchmark_configurations",
            r"""
\begin{table}[htbp]
\fontsize{8}{10}\selectfont
\footnotesize
\caption{\small Benchmark applications used for validation with their respective input parameter values. MB and CB denote memory-bound and compute-bound, respectively.}\label{table:validation-benchmarks}
\centering
\setlength\extrarowheight{2pt}
% \rowcolors{3}{white}{gray!20}
{\renewcommand{\arraystretch}{1.5}%
\begin{tabularx}{0.9\textwidth}{z|s|zX}
% ---- paste here
\rowcolor{gray!10} Benchmark
& Type 
& \multicolumn{2}{c}{Input parameters} \\ \hline
\multirow[t]{2}{*}{\shortstack[l]{Vectoradd}} 
& \multirow[t]{2}{*}{\shortstack[l]{MB}} 
& \textbf{dtype} & 32, 64 \\ 
& & \textbf{length} & 100, 1000, 10000, 20000, 500000 \\ \hline
%
\multirow[t]{4}{*}{\shortstack[l]{Matrixmul (naive)}}
& \multirow[t]{4}{*}{\shortstack[l]{CB}} 
& \textbf{dtype} & 32 \\
&  & \textbf{m} & 32, 64, 128, 512 \\
&  & \textbf{n} & 32, 64, 128, 512 \\
&  & \textbf{p} & 32, 64, 128, 512 \\ \hline
%
\multirow[t]{2}{*}{\shortstack[l]{Matrixmul}}
& \multirow[t]{2}{*}{\shortstack[l]{CB}}
& \textbf{dtype} & 32 \\
&  & \textbf{rows} & 32, 64, 128, 256, 512 \\ \hline
%
\multirow[t]{2}{*}{\shortstack[l]{Transpose}}
& \multirow[t]{2}{*}{\shortstack[l]{MB}}
& \textbf{dim} & 128, 256, 512 \\
&  & \textbf{variant} & coalesced, naive \\ \hline
%
\multirow[t]{1}{*}{\shortstack[l]{Babelstream}}
& \multirow[t]{1}{*}{\shortstack[l]{MB}}
& \textbf{size} & 1024, 10240, 102400 \\ \hline
%
% ---- paste end
\end{tabularx}}
\end{table}
""",
        ),
    ]

    for name, tex in reversed(equations):
        tex_code = r"""
\documentclass[preview]{standalone}
"""
        tex_code += utils.TEX_PACKAGES
        tex_code += r"""
\begin{document}
"""
        tex_code += tex
        tex_code += r"""
\end{document}
        """

        assert isinstance(tex_code, str)
        print(tex_code)
        pdf_output_path = (plot.EQUATIONS_DIR / name).with_suffix(".pdf")
        try:
            utils.render_latex(tex_code, output_path=pdf_output_path)
            pass
        except Exception as e:
            print(tex_code)
            print("##################")
            raise ValueError(tex_code)
            # return
            raise e

        print(color("wrote {}".format(pdf_output_path), fg="cyan"))

        if png:
            png_output_path = (plot.EQUATIONS_DIR / "png" / name).with_suffix(".png")
            utils.convert_to_png(input_path=pdf_output_path, output_path=png_output_path, density=600)
            print(color("wrote {}".format(png_output_path), fg="cyan"))

    pass


if __name__ == "__main__":
    main()
