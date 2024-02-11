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
def equations():
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
    ]

    for name, tex in equations:
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

        png_output_path = (plot.EQUATIONS_DIR / "png" / name).with_suffix(".png")
        utils.convert_to_png(
            input_path=pdf_output_path, output_path=png_output_path, density=600
        )
        print(color("wrote {}".format(png_output_path), fg="cyan"))

    pass


if __name__ == "__main__":
    main()
