import typing
import pandas as pd
import numpy as np
from wasabi import color
from copy import copy
from pprint import pprint
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from gpucachesim import REPO_ROOT_DIR
import gpucachesim.tex as tex
import gpucachesim.utils as utils
import gpucachesim.plot as plot
import gpucachesim.tex as tex

RESULTS_DIR = REPO_ROOT_DIR / "results"
assert RESULTS_DIR.is_dir()

root_dir = Path("/home/roman/dev/simulators")
assert root_dir.is_dir()
benchmark_dir = root_dir / "benchmarks"
run_dir = root_dir / "run"
assert benchmark_dir.is_dir()
assert run_dir.is_dir()

import sys

sys.path.insert(0, str(root_dir))
import gpusims
import gpusims.plot.metrics as metric
import gpucachesim.stats.metrics as metric_funcs
from gpusims.plot.data import PlotData
from gpusims.config import Config, parse_configs
from gpusims.bench import parse_benchmarks

pd.set_option("display.max_rows", 700)
pd.set_option("display.max_columns", 700)
np.seterr(all="raise")

# define ordering that makes sense (e.g. hw and accel close)
SELECTED_SIMULATORS = [
    gpusims.TEJAS,
    gpusims.MACSIM,
    gpusims.MULTI2SIM,
    gpusims.ACCELSIM_PTX,
    gpusims.ACCELSIM_SASS,
    gpusims.NATIVE,
]

SIM_TEX_NAME = {
    gpusims.TEJAS: r"\textsc{GpuTejas}",
    gpusims.MACSIM: r"\textsc{MacSim}",
    gpusims.MULTI2SIM: r"\textsc{Multi2Sim}",
    gpusims.NATIVE: r"\textsc{Native}",
    gpusims.ACCELSIM_PTX: r"\textsc{AccelSim} {\tiny (PTX)}",
    gpusims.ACCELSIM_SASS: r"\textsc{AccelSim} {\tiny (SASS)}",
}

SIM_COLORS = {
    gpusims.TEJAS: plot.hex_to_rgb("#7b2cbf"),
    gpusims.MACSIM: plot.hex_to_rgb("#ffddd2"),
    gpusims.MULTI2SIM: plot.hex_to_rgb("#0766ad"),
    gpusims.NATIVE: plot.hex_to_rgb("#f3f3f3"),
    gpusims.ACCELSIM_PTX: plot.hex_to_rgb("#c5e898"),
    gpusims.ACCELSIM_SASS: plot.hex_to_rgb("#29adb2"),
}

SIM_COLORS = {
    gpusims.TEJAS: "f3f9d2",
    gpusims.MACSIM: "cbeaa6",
    gpusims.MULTI2SIM: "bdd9bf",
    gpusims.NATIVE: "000000",
    gpusims.ACCELSIM_PTX: "c0d684",
    gpusims.ACCELSIM_SASS: "c0d684",
}
SIM_COLORS = {k: "29adb2" for k in SIM_COLORS.keys()}
SIM_COLORS = {k: plot.hex_to_rgb(v) for k, v in SIM_COLORS.items()}


# define ordering of inputs that makes sense
SELECTED_BENCHMARKS = [
    (
        "babelstream",
        [
            ("--arraysize 1024 --numtimes 1", "1024"),
            ("--arraysize 10240 --numtimes 1", "10240"),
            ("--arraysize 102400 --numtimes 1", "102400"),
            # ("--arraysize 1024 --numtimes 2", "1024 (2x)"),
        ],
    ),
    (
        "vectoradd",
        [
            # [inp.args for inp in benchmarks["vectoradd"].inputs]),
            ("1000", "1K"),
            ("1000000", "1M"),
        ],
    ),
    (
        "cuda4-matrixmul",
        [
            # [inp.args for inp in benchmarks["cuda4-matrixmul"].inputs]),
            ("32", "32x32"),
            ("128", "128x128"),
            ("512", "512x512"),
        ],
    ),
    (
        "cuda10-matrixmul",
        [
            ("-wA=32 -hA=32 -wB=32 -hB=32", "32x32 32x32"),
            ("-wA=128 -hA=128 -wB=128 -hB=128", "128x128 128x128"),
            ("-wA=512 -hA=512 -wB=512 -hB=512", "512x512 512x512"),
            # ("-wA=32 -hA=64 -wB=64 -hB=32", "32x64 64x32"),
        ],
    ),
    (
        "cuda6-transpose",
        [
            ("-repeat=1 -dimX=32 -dimY=32", "32x32"),
            ("-repeat=1 -dimX=64 -dimY=64", "64x64"),
            ("-repeat=1 -dimX=128 -dimY=128", "128x128"),
            # ("-repeat=3 -dimX=32 -dimY=32", "32x32 (3x)"),
        ],
    ),
    (
        "cuda10-transpose",
        [
            ("-repeat=1 -dimX=32 -dimY=32", "32x32"),
            ("-repeat=1 -dimX=64 -dimY=64", "64x64"),
            ("-repeat=1 -dimX=128 -dimY=128", "128x128"),
            # ("-repeat=3 -dimX=32 -dimY=32", "32x32"),
        ],
    ),
]


def dedup_ordered(l: list[typing.Any]) -> list[typing.Any]:
    return list(dict.fromkeys(l))


def build_metric_df(
    metric_cls,
    simulators,
    benchmarks,
    configs,
):
    all_metric_df = []
    for config in configs:
        for bench, selected_bench_inputs in benchmarks:
            for inp_args, _ in selected_bench_inputs:

                inp_matches = [i for i in bench.inputs if i.args.strip() == inp_args.strip()]
                assert len(inp_matches) == 1
                inp = inp_matches[0]
                print(config.name, bench.name, inp)

                plot_data = PlotData(benchmark=bench, config=config, inp=inp)

                for sim in simulators:
                    if not bench.enabled(sim.ID):
                        continue
                    if not inp.enabled(sim.ID):
                        continue

                    bench_config = sim(
                        run_dir=run_dir / sim.ID.lower(),
                        benchmark=bench,
                        config=config,
                    )
                    if not bench_config.input_path(inp).is_dir():
                        raise ValueError(f"WARN: {bench_config.input_path(inp)} does not exist")

                    plot_data[sim.ID] = bench_config.load_dataframe(inp)

                metric = metric_cls(plot_data)
                metric_df = metric.compute()
                metric_df["Benchmark"] = f"{bench.name}<br>{inp.args}"
                metric_df["Config"] = config.key
                all_metric_df.append(metric_df)

    all_metric_df = pd.concat(all_metric_df)
    return all_metric_df


def build_result_table(aggregated, configs, simulators):
    table = ""

    header_simulators = [""]
    header_configs = [""]

    simulators = [s for s in simulators if s.ID != gpusims.NATIVE]

    for i, sim in enumerate(simulators):
        is_last = i == len(simulators) - 1

        sim_col = r"\multicolumn{2}{" + ("c" if is_last else "c|") + "}{"
        # sim_col += sim.TEX_NAME
        sim_col += SIM_TEX_NAME[sim.ID]
        sim_col += "}"

        header_simulators.append(sim_col)
        for c in configs:
            config_col = r"\scriptsize \centering\arraybackslash {}".format(c.name)
            header_configs.append(config_col)

    table += "\n & ".join(header_simulators) + r" \\" + "\n"
    table += " % "
    table += "\n & ".join(header_configs) + r" \\ \hline" + "\n"
    table += " %\n"

    for metric_key, metric_name, is_percent in [
        ("corr", "Corr.", False),
        ("mape", "MAPE", True),
        ("rmspe", "RMSPE", True),
    ]:
        line = [metric_name]
        for sim in simulators:
            for config in configs:
                value = aggregated.loc[(config.key, sim.ID), metric_key]
                if is_percent:
                    value *= 100
                    value = r"${value:.{precision}f}\%$".format(value=value, precision=1 if abs(value) < 100.0 else 0)
                else:
                    value = r"${value:.{precision}f}$".format(value=value, precision=3 if value > 0.0 else 2)
                line.append(value)
        table += " & ".join(line) + r" \\" + "\n"
        table += " %\n"

    table += r"\hline"

    return table


def result_table(force=False, verbose=False, png=False, png_density=600):
    simulators = copy(gpusims.SIMULATORS)
    configs = parse_configs(benchmark_dir / "configs" / "configs.yml")
    benchmarks = parse_benchmarks(benchmark_dir / "benchmarks.yml")

    # define ordering that makes sense
    # sm86_a4000 is so close to the rtx3070 we exclude it?
    selected_configs = ["sm6_gtx1080", "sm86_rtx3070"]

    selected_simulators = [simulators[s] for s in SELECTED_SIMULATORS]
    selected_simulators = dedup_ordered(selected_simulators)

    selected_configs = [configs[c] for c in selected_configs]
    selected_configs = dedup_ordered(selected_configs)

    pprint(benchmarks)
    pprint(selected_simulators)
    pprint(selected_configs)

    cycles_df_csv_path = RESULTS_DIR / "compat_cycles.csv"
    if force or not cycles_df_csv_path.is_file():
        print("generating", cycles_df_csv_path)
        selected_benchmarks = [(benchmarks[b], inputs) for b, inputs in SELECTED_BENCHMARKS]
        cycles_df = build_metric_df(
            metric_cls=gpusims.plot.metrics.Cycles,
            simulators=selected_simulators,
            benchmarks=selected_benchmarks,
            configs=selected_configs,
        )
        cycles_df.to_csv(cycles_df_csv_path, index=False)
    else:
        cycles_df = pd.read_csv(
            cycles_df_csv_path,
            header=0,
        )

    unique_benchmarks = sorted(list(cycles_df["Benchmark"].unique()))
    cycles_df["Benchmark"] = cycles_df["Benchmark"].apply(lambda b: unique_benchmarks.index(b))
    cycles_df = cycles_df.set_index(["Config", "Simulator", "Benchmark"])

    native_values = cycles_df.loc[pd.IndexSlice[:, "native", :], :]
    native_values = native_values.rename(columns={"Value": "native"})
    joined = cycles_df.reset_index().merge(
        native_values.reset_index(),
        on=["Config", "Benchmark"],
        how="left",
        suffixes=("", "_drop"),
        sort=False,
    )
    joined = joined.drop(columns=[col for col in joined if col.endswith("_drop")])
    joined = joined.set_index(["Config", "Simulator", "Benchmark"])

    # just a preview of the data with some errors
    preview_df = joined.copy()
    mape = metric_funcs.mape(
        true_values=preview_df["native"],
        values=preview_df["Value"],
        raw=True,
    )
    smape = metric_funcs.smape(
        true_values=preview_df["native"],
        values=preview_df["Value"],
        raw=True,
    )

    preview_df["smape"] = smape
    preview_df["mape"] = mape
    # print(preview_df)

    # preview_cols = ["Config", "Simulator", "bench_preview", "Value", "native"]
    # print(joined.reset_index()[preview_cols])

    grouped = joined.groupby(["Config", "Simulator"], dropna=False, sort=False)

    aggregated = grouped.median()

    aggregated["mape"] = grouped.apply(lambda df: metric_funcs.mape(true_values=df["native"], values=df["Value"]))
    aggregated["rmspe"] = grouped.apply(lambda df: metric_funcs.rmspe(true_values=df["native"], values=df["Value"]))
    aggregated["corr"] = grouped.apply(
        lambda df: metric_funcs.correlation(true_values=df["native"], values=df["Value"])
    )
    print(aggregated)

    table = build_result_table(
        aggregated=aggregated,
        simulators=selected_simulators,
        configs=selected_configs,
    )

    tex_document_code = r"""
\documentclass[preview]{standalone}
"""
    tex_document_code += tex.TEX_PACKAGES
    tex_document_code += r"""
\begin{document}
"""

    tex_code = r"\begin{tabularx}{\textwidth}{" + "\n"
    tex_code += tex.r(width="10mm") + "\n"

    # tex_code += "|zz|zz|zz|zz|zz}"
    base_col = tex.r()
    config_cols = "|" + (base_col * len(selected_configs)) + "\n"
    tex_code += config_cols * len([s for s in selected_simulators if s.ID != gpusims.NATIVE])
    tex_code += "}"
    tex_code += table
    tex_code += r"""
%
\end{tabularx}
"""

    tex_document_code += r"""
\begin{table}[tbh!]
\fontsize{8}{10}\selectfont
\footnotesize
\centering
{\renewcommand{\arraystretch}{1.5}%
"""
    tex_document_code += tex_code
    tex_document_code += r"""
}
\end{table}
\end{document}
"""

    print(tex_code)

    # write latex
    filename = "compat_empirical_result_table"
    tex_output_path = (plot.TABLE_DIR / filename).with_suffix(".tex")
    with open(tex_output_path, "w") as f:
        f.write(tex_code)
    print(color("wrote {}".format(tex_output_path), fg="cyan"))

    pdf_output_path = (plot.TABLE_DIR / filename).with_suffix(".pdf")
    try:
        tex.render_latex(tex_document_code, output_path=pdf_output_path)
    except Exception as e:
        print(tex_document_code)
        raise e
    print(color("wrote {}".format(pdf_output_path), fg="cyan"))


def slowdowns(force=False, verbose=False, png=False, png_density=600):
    simulators = copy(gpusims.SIMULATORS)
    configs = parse_configs(benchmark_dir / "configs" / "configs.yml")
    benchmarks = parse_benchmarks(benchmark_dir / "benchmarks.yml")

    # define ordering that makes sense
    # sm86_a4000 is so close to the rtx3070 we exclude it
    selected_configs = ["sm6_gtx1080", "sm86_rtx3070"]

    selected_simulators = [simulators[s] for s in SELECTED_SIMULATORS]
    selected_simulators = dedup_ordered(selected_simulators)

    selected_configs = [configs[c] for c in selected_configs]
    selected_configs = dedup_ordered(selected_configs)

    pprint(benchmarks)
    pprint(selected_simulators)
    pprint(selected_configs)

    exec_time_df_csv_path = RESULTS_DIR / "compat_exec_time.csv"
    if force or not exec_time_df_csv_path.is_file():
        print("generating", exec_time_df_csv_path)
        selected_benchmarks = [(benchmarks[b], inputs) for b, inputs in SELECTED_BENCHMARKS]
        exec_time_df = build_metric_df(
            metric_cls=gpusims.plot.metrics.ExecutionTime,
            simulators=selected_simulators,
            benchmarks=selected_benchmarks,
            configs=selected_configs,
        )
        exec_time_df.to_csv(exec_time_df_csv_path, index=False)
    else:
        exec_time_df = pd.read_csv(
            exec_time_df_csv_path,
            header=0,
        )

    unique_benchmarks = sorted(list(exec_time_df["Benchmark"].unique()))
    exec_time_df["Benchmark"] = exec_time_df["Benchmark"].apply(lambda b: unique_benchmarks.index(b))
    # exec_time_df = exec_time_df.set_index(["Config", "Simulator", "Benchmark"])

    # split sim and trace values into columns
    exec_time_df = exec_time_df.pivot(
        index=["Config", "Simulator", "Benchmark"],
        columns="Kind",
        values="Value",
    )

    exec_time_df.loc[exec_time_df["Sim"] == 0.0, ["Sim", "Trace"]] = np.nan

    print(exec_time_df)
    print(exec_time_df.index)
    print(exec_time_df.columns)

    native_values = exec_time_df.loc[pd.IndexSlice[:, "native", :], :]
    native_values = native_values.rename(columns={"Sim": "Native"})
    assert (native_values["Trace"] == 0.0).all()
    native_values = native_values.drop(columns=["Trace"])

    joined = exec_time_df.reset_index().merge(
        native_values.reset_index(),
        on=["Config", "Benchmark"],
        how="left",
        suffixes=("", "_drop"),
        sort=False,
    )
    joined = joined.drop(columns=[col for col in joined if col.endswith("_drop")])
    joined = joined.set_index(["Config", "Simulator", "Benchmark"])

    joined["Slowdown"] = metric_funcs.slowdown(baseline=joined["Native"], values=joined["Sim"])

    print(joined)

    grouped = joined.groupby(["Simulator"], dropna=False, sort=False)
    aggregated = grouped["Slowdown"].aggregate(["median", "mean", "std"])
    aggregated["norm_std_mean"] = aggregated["std"] / aggregated["mean"]
    aggregated["norm_std_median"] = aggregated["std"] / aggregated["median"]
    print(aggregated)

    # sort by mean slowdown
    aggregated = aggregated.sort_values(by=["mean"])

    fontsize = plot.FONT_SIZE_PT - 3
    font_family = "Helvetica"

    bar_width = 10
    spacing = 2
    group_spacing = 1 * bar_width

    # use the sorted simulator order
    plot_simulators = aggregated.index.get_level_values("Simulator")
    plot_simulators = [simulators[s] for s in plot_simulators if s != gpusims.NATIVE]
    # [s for s in selected_simulators if s.ID != gpusims.NATIVE]
    # plot_simulators = [s for s in selected_simulators if s.ID != gpusims.NATIVE]
    # group_width = len(plot_simulators) * (bar_width + spacing) + group_spacing
    group_width = 1 * (bar_width + spacing) + group_spacing

    ylabel = "Mean slowdown"

    figsize = (0.7 * plot.DINA4_WIDTH_INCHES, 0.13 * plot.DINA4_HEIGHT_INCHES)

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})
    plt.rcParams["hatch.linewidth"] = 0.5
    matplotlib.rcParams["hatch.linewidth"] = 0.5

    fig = plt.figure(
        figsize=figsize,
        layout="constrained",
    )
    ax = plt.axes()

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=1,
        color="black",
        alpha=0.1,
        zorder=1,
    )

    # we chosee the MEAN
    agg_col = "mean"

    for sim_idx, sim in enumerate(plot_simulators):
        idx = sim_idx * group_width + (0.0 + 0.5) * (bar_width + spacing)

        x = [idx]
        y = aggregated.loc[sim.ID, agg_col]

        bar_color = plot.plt_rgba(*SIM_COLORS[sim.ID])
        print(bar_color)
        ax.bar(
            x,
            y,
            color=bar_color,
            hatch=None,
            width=bar_width,
            linewidth=1,
            edgecolor="black",
            zorder=2,
            label=sim.NAME,
        )

    ax.set_ylabel(ylabel)
    ax.axes.set_zorder(10)

    ymax = aggregated.loc[[s.ID for s in plot_simulators], agg_col].max()
    print("ymax is", ymax)
    num_yticks = 6

    ymax = max(1.2 * ymax, 1)
    ax.set_ylim(0, ymax)

    yticks, min_precision = plot.linear_range_with_power_step_size(min=0, max=ymax, num_ticks=num_yticks, base=10)
    ytick_labels = [plot.human_format_thousands(v, round_to=min_precision, variable_precision=True) for v in yticks]
    print("ytick labels", ytick_labels)

    xtick_labels = [s.NAME.replace(" ", "\n") for s in plot_simulators]
    print("xtick labels", xtick_labels)

    xticks = np.arange(0, len(xtick_labels), dtype=np.float64)
    xticks *= group_width
    xticks += 0.5 * float((group_width - group_spacing))
    xmargin = 0.5 * group_spacing
    ax.set_xlim(-xmargin, len(xtick_labels) * group_width - xmargin)

    ax.set_yticks(yticks, ytick_labels)
    ax.set_xticks(
        xticks,
        xtick_labels,
        rotation=0,
    )

    plot_dir = plot.PLOT_DIR
    filename = "compat_empirical_slowdowns"
    pdf_output_path = (plot_dir / filename).with_suffix(".pdf")
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_output_path)
    print(color("wrote {}".format(pdf_output_path), fg="cyan"))

    if png:
        png_output_path = (plot_dir / "png" / filename).with_suffix(".png")
        utils.convert_to_png(
            input_path=pdf_output_path,
            output_path=png_output_path,
            density=png_density,
        )
